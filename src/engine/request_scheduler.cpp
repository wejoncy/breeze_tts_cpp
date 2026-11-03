#include "engine/request_scheduler.h"
#include "engine/request_executor.h"
#include "execution/model_runner.h"
#include "runtime/ordered_mailbox.h"
#include "support/log.h"
#include "support/profiler.h"
#include <thread>
#include <deque>
#include <map>
#include <unordered_map>
#include <chrono>
#include <type_traits>

namespace tts {
using Clock = std::chrono::steady_clock;
namespace {
// Synchronous APIs wait on a reply queue; submission only posts its command.
template <class T> struct Reply {
    struct Message {
        std::conditional_t<std::is_void_v<T>, bool, T> value{};
        std::exception_ptr error;
    };
    Mailbox<Message> ready{1};
    void fail(std::exception_ptr error) {
        send(ready, Message{{}, std::move(error)});
    }
    template <class F> void run(F &&fn) noexcept {
        Message message;
        try {
            if constexpr (std::is_void_v<T>)
                fn();
            else
                message.value = fn();
        } catch (...) {
            message.error = std::current_exception();
        }
        send(ready, std::move(message));
    }
    T get() {
        Message message;
        ready.wait_dequeue(message);
        if (message.error)
            std::rethrow_exception(message.error);
        if constexpr (!std::is_void_v<T>)
            return std::move(message.value);
    }
};
struct Result {
    GenerationFrame frame;
    std::exception_ptr error;
    size_t bytes = 0;
};
struct Job {
    std::string request_id;
    std::deque<Result> results;
    std::function<void(Result)> reader;
    Clock::time_point deadline = Clock::time_point::max();
    std::vector<Input> requests;
    GenerationOptions options;
    uint64_t execution_id = 0;
    std::unique_ptr<RequestPlan> plan;
    std::unique_ptr<RequestPreparation> preparation;
    std::unique_ptr<PreparationTask> task;
    Stats stats;
    std::exception_ptr error;
    size_t buffered = 0, next_lane = 0, completed = 0, buffer_limit = 0;
    std::deque<size_t> waiting_inputs;
    uint64_t retry_epoch = 0;
    bool done = false, cancelled = false;
    bool registered = false, released = false;
    Clock::time_point submitted = Clock::now();
    double elapsed() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - submitted).count();
    }
    bool aborted() const {
        return cancelled || error;
    }
};
} // namespace
struct RequestScheduler::Impl {
    struct Command {
        enum class Kind { Run, Add, Cancel, Shutdown, Prepared, Planned };
        Kind kind = Kind::Run;
        std::function<void()> run;
        std::string request_id;
        std::unique_ptr<Job> job;
        std::unique_ptr<PreparationTask> preparation;
        std::function<void(std::unique_ptr<PreparationTask>, std::exception_ptr)> prepared;
        std::exception_ptr preparation_error;
        uint64_t owner = 0;
    };
    struct Active {
        ExecutionRow row;
        std::shared_ptr<Job> job;
        Stats previous;
        bool prefilled = false;
    };
    RequestPlanner &planner;
    int capacity;
    double frame_period_ms, last_decode_ms = 0;
    uint64_t engine_id, lane_sequence = 0, sequence = 0;
    OrderedMailbox<Command> incoming;
    // Admission transfers one slot through this queue; it shares no counter.
    Mailbox<bool> available;
    // The CPU scheduler exclusively owns the mutable scheduling state below.
    std::unordered_map<std::string, std::shared_ptr<Job>> jobs;
    std::deque<std::shared_ptr<Job>> planning, preparing, pending;
    std::deque<Command> prepared_results;
    std::unordered_map<uint64_t, std::shared_ptr<Job>> inflight_preparations;
    std::vector<Active> active;
    std::vector<uint64_t> releases;
    std::vector<uint64_t> request_releases;
    std::deque<std::function<void(std::exception_ptr)>> controls;
    size_t request_limit = 32, issued_slots = 32, buffer_limit = 1 << 20, control_count = 0;
    bool stopping = false, busy = false, preparation_busy = false;
    // One executing batch and one queued successor overlap CPU scheduling with
    // GPU work. Result ownership remains with the scheduler, in queue order.
    static constexpr size_t pipeline_depth = 2;
    size_t inflight_batches = 0;
    std::vector<Completion> output_pool;
    uint64_t resource_epoch = 1;
    std::exception_ptr executor_error;
    // The caller owns lifecycle operations and stops ingress before shutdown.
    enum class State { Ready, Running, Stopped };
    State state = State::Ready;
    RequestExecutor executor;
    std::thread scheduler;
    Impl(RequestPlanner &p, ModelRunner &r, int batch, double period, uint64_t id)
        : planner(p), capacity(batch), frame_period_ms(period), engine_id(id),
          executor(r, batch, id, [this](Completion result) {
              send(incoming,
                   Command{Command::Kind::Run, [this, result = std::move(result)]() mutable {
                               complete(result);
                               if (result.executed)
                                   output_pool.push_back(std::move(result));
                           }});
          }) {
        const size_t samples = r.max_step_samples();
        if (!samples)
            throw std::logic_error("model defines an empty audio step capacity");
        output_pool.resize(pipeline_depth);
        for (auto &output : output_pool) {
            output.rows.resize(size_t(capacity));
            for (auto &row : output.rows) {
                // Touch pages during initialization, before execution starts.
                row.pcm.resize(samples);
                row.pcm.clear();
            }
        }
        for (size_t i = 0; i < issued_slots; ++i)
            send(available, true);
    }
    ~Impl() {
        stop();
    }
    std::runtime_error unavailable() const {
        return std::runtime_error(state == State::Stopped
                                      ? "engine is shut down"
                                      : "engine is not running; call run_async() first");
    }
    void start() {
        if (state == State::Running)
            return;
        if (state == State::Stopped)
            throw unavailable();
        try {
            executor.start();
            scheduler = std::thread([this] { schedule(); });
            state = State::Running;
        } catch (...) {
            executor.stop();
            state = State::Stopped;
            throw;
        }
    }
    void post(Command command) {
        if (state != State::Running)
            throw unavailable();
        send(incoming, std::move(command));
    }
    template <class F> void post(F fn) {
        post(Command{Command::Kind::Run, std::move(fn)});
    }
    template <class F> auto call(F fn) -> std::invoke_result_t<F> {
        Reply<std::invoke_result_t<F>> reply;
        post([this, fn = std::move(fn), &reply]() mutable {
            reply.run([&]() -> std::invoke_result_t<F> {
                if (stopping)
                    throw std::runtime_error("engine is shutting down");
                return fn();
            });
        });
        return reply.get();
    }
    void check_running() const {
        if (stopping)
            throw std::runtime_error("engine is shutting down");
        if (executor_error)
            std::rethrow_exception(executor_error);
    }
    void release_slot() {
        if (issued_slots > request_limit)
            --issued_slots;
        else
            send(available, true);
    }
    void configure(size_t limit, size_t output_bytes) {
        bool slot;
        while (available.try_dequeue(slot))
            --issued_slots;
        request_limit = limit;
        buffer_limit = output_bytes;
        while (issued_slots < request_limit) {
            send(available, true);
            ++issued_slots;
        }
    }
    void unregister(std::shared_ptr<Job> job) {
        if (!job->registered)
            return;
        jobs.erase(job->request_id);
        job->registered = false;
        if (job->released)
            release_slot();
    }
    void retire(const std::shared_ptr<Job> &job) {
        if (job->released || inflight_preparations.count(job->execution_id) ||
            std::find(planning.begin(), planning.end(), job) != planning.end() ||
            std::find(preparing.begin(), preparing.end(), job) != preparing.end() ||
            std::find(pending.begin(), pending.end(), job) != pending.end() ||
            std::any_of(active.begin(), active.end(),
                        [&](const Active &a) { return a.job == job; }))
            return;
        job->released = true;
        request_releases.push_back(job->execution_id);
        job->plan.reset();
        job->preparation.reset();
        job->task.reset();
        if (!job->registered)
            release_slot();
        log::Scope scope(engine_id, job->execution_id);
        TTS_LOG(TTS_LOG_INFO, "request", "complete batch={} frames={} total_ms={:.3f} cancelled={}",
                job->requests.size(), job->stats.frames, job->elapsed(), job->cancelled);
    }
    void stop() {
        if (state == State::Stopped)
            return;
        if (state == State::Running) {
            send(incoming, Command{Command::Kind::Shutdown});
            scheduler.join();
        }
        state = State::Stopped;
    }
    void stop_work() {
        stopping = true;
        for (auto it = jobs.begin(); it != jobs.end();) {
            auto job = (it++)->second;
            cancel(job);
        }
        auto error = std::make_exception_ptr(std::runtime_error("engine is shutting down"));
        for (auto &control : controls)
            control(error);
        controls.clear();
        jobs.clear();
        planning.clear();
        preparing.clear();
        pending.clear();
        executor.stop();
        inflight_preparations.clear();
        prepared_results.clear();
        active.clear();
        releases.clear();
        request_releases.clear();
    }
    std::unique_ptr<Job> snapshot(const Request &request) {
        auto job = std::make_unique<Job>();
        job->request_id = request.request_id;
        job->requests = request.inputs;
        job->options = request.options;
        bool slot;
        if (!available.try_dequeue(slot))
            throw RequestOverloaded("request queue is full");
        return job;
    }
    void enqueue(std::unique_ptr<Job> incoming_job) {
        std::shared_ptr<Job> job(std::move(incoming_job));
        auto [it, inserted] = jobs.emplace(job->request_id, job);
        if (!inserted) {
            release_slot();
            // Results are addressed only by request ID. A rejected duplicate
            // must not insert an error into the original request's output stream.
            TTS_LOG(TTS_LOG_WARN, "request", "ignored duplicate request_id={}", job->request_id);
            return;
        }
        job->registered = true;
        job->execution_id = ++sequence;
        job->buffer_limit = buffer_limit;
        try {
            check_running();
            planning.push_back(job);
        } catch (...) {
            finish(job, std::current_exception());
            retire(job);
        }
    }
    void emit(const std::shared_ptr<Job> &job, Result result) {
        result.frame.request_id = job->request_id;
        if (!job->reader) {
            job->buffered += result.bytes;
            job->results.push_back(std::move(result));
            return;
        }
        auto reader = std::move(job->reader);
        job->deadline = Clock::time_point::max();
        if (result.frame.done) {
            retire(job);
            unregister(job);
        }
        reader(std::move(result));
    }
    void finish(const std::shared_ptr<Job> &job, std::exception_ptr failure = {}) {
        if (job->done)
            return;
        job->error = failure;
        job->done = true;
        job->stats.total_ms = job->elapsed();
        job->results.clear();
        job->buffered = 0;
        Result result;
        result.error = failure;
        result.frame.stats = job->stats;
        result.frame.done = true;
        result.frame.cancelled = job->cancelled;
        emit(job, std::move(result));
    }
    void cancel(std::shared_ptr<Job> job) {
        job->cancelled = job->done = true;
        job->results.clear();
        job->buffered = 0;
        Result result;
        result.frame.done = result.frame.cancelled = true;
        result.frame.stats = job->stats;
        emit(job, std::move(result));
        unregister(job);
        for (auto *queue : {&planning, &preparing, &pending})
            queue->erase(std::remove(queue->begin(), queue->end(), job), queue->end());
        retire(job);
    }
    void publish(const std::shared_ptr<Job> &job, GenerationFrame frame) {
        if (job->cancelled || job->done)
            return;
        if (frame.chunks.empty() && job->completed != job->requests.size())
            return;
        Result result;
        for (const auto &chunk : frame.chunks)
            result.bytes += chunk.second.size() * sizeof(int16_t);
        if (!job->reader && result.bytes > job->buffer_limit - job->buffered) {
            finish(job, std::make_exception_ptr(
                            std::runtime_error("audio consumer exceeded its buffer limit")));
            return;
        }
        if (!job->stats.first_audio_ms && result.bytes)
            job->stats.first_audio_ms = job->elapsed();
        job->stats.total_ms = job->elapsed();
        frame.stats = job->stats;
        frame.done = job->completed == job->requests.size();
        job->done = frame.done;
        result.frame = std::move(frame);
        emit(job, std::move(result));
    }
    void read(const std::string &id, int timeout_ms, Reply<GenerationFrame> &reply) {
        auto it = jobs.find(id);
        if (it == jobs.end())
            throw std::invalid_argument("unknown request_id");
        auto job = it->second;
        if (job->reader)
            throw std::logic_error("request already has a reader");
        job->reader = [&reply](Result result) {
            reply.run([&] {
                if (result.error)
                    std::rethrow_exception(result.error);
                return std::move(result.frame);
            });
        };
        if (!job->results.empty()) {
            auto result = std::move(job->results.front());
            job->results.pop_front();
            job->buffered -= result.bytes;
            emit(job, std::move(result));
        } else if (timeout_ms == 0) {
            Result result;
            result.frame.ready = false;
            emit(job, std::move(result));
        } else if (timeout_ms > 0) {
            job->deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
        }
    }
    Clock::time_point expire_reads() {
        auto now = Clock::now(), next = Clock::time_point::max();
        for (auto &[id, job] : jobs) {
            if (!job->reader)
                continue;
            if (job->deadline <= now) {
                Result result;
                result.frame.ready = false;
                emit(job, std::move(result));
            } else {
                next = std::min(next, job->deadline);
            }
        }
        return next;
    }
    void continue_planning(const std::shared_ptr<Job> &job) {
        job->task = job->preparation->next();
        if (job->task) {
            preparing.push_back(job);
        } else {
            job->plan = job->preparation->finish();
            job->preparation.reset();
            pending.push_back(job);
        }
    }
    void plan_ready() {
        // Yield to queue events after each CPU plan, keeping the executor fed
        // while other requests wait for GPU conditioning.
        if (!prepared_results.empty()) {
            auto result = std::move(prepared_results.front());
            prepared_results.pop_front();
            planned(std::move(result));
            return;
        }
        if (!planning.empty()) {
            Range trace("tts.engine.plan");
            auto job = planning.front();
            planning.pop_front();
            if (job->aborted()) { retire(job); return; }
            try {
                job->preparation = planner.begin(job->requests, job->options);
                continue_planning(job);
            } catch (...) {
                finish(job, std::current_exception());
                retire(job);
            }
        }
    }
    void planned(Command command) {
        Range trace("tts.engine.prepare_complete");
        auto found = inflight_preparations.find(command.owner);
        if (found != inflight_preparations.end()) {
            auto job = std::move(found->second);
            inflight_preparations.erase(found);
            if (job->aborted()) retire(job);
            else try {
                if (command.preparation_error)
                    std::rethrow_exception(command.preparation_error);
                job->preparation->accept(std::move(command.preparation));
                continue_planning(job);
            } catch (...) {
                finish(job, std::current_exception());
                retire(job);
            }
        }
        preparation_busy = !inflight_preparations.empty();
    }
    bool dispatch_preparation() {
        std::vector<PreparedTask> tasks;
        std::vector<uint64_t> owners;
        const size_t available_rows = size_t(capacity) - active.size();
        while (!preparing.empty() && tasks.size() < available_rows) {
            auto job = preparing.front();
            preparing.pop_front();
            if (job->aborted()) { retire(job); continue; }
            tasks.push_back({std::move(job->task), {}});
            owners.push_back(job->execution_id);
            inflight_preparations.emplace(job->execution_id, job);
        }
        if (tasks.empty()) return false;
        preparation_busy = true;
        executor.prepare_tasks(std::move(tasks), [this, owners = std::move(owners)](size_t index, PreparedTask result) {
            Command command;
            command.kind = Command::Kind::Planned;
            command.owner = owners[index];
            command.preparation = std::move(result.task);
            command.preparation_error = result.error;
            send(incoming, std::move(command));
        });
        return true;
    }
    void dispatch() {
        if (busy || inflight_batches == pipeline_depth || stopping || executor_error)
            return;
        Range trace("tts.engine.dispatch");
        for (auto it = pending.begin(); it != pending.end();) {
            if (!(*it)->aborted()) {
                ++it;
                continue;
            }
            auto job = *it;
            it = pending.erase(it);
            finish(job, job->error);
            retire(job);
        }
        for (auto it = active.begin(); it != active.end();) {
            auto job = it->job;
            if (!job->aborted()) {
                ++it;
                continue;
            }
            releases.push_back(it->row.id);
            it = active.erase(it);
            finish(job, job->error);
            retire(job);
        }
        if (!controls.empty()) {
            if (inflight_batches || preparation_busy)
                return;
            auto control = std::move(controls.front());
            controls.pop_front();
            busy = true;
            executor.control(std::move(control));
            return;
        }
        // Existing prefill checkpoints retain their rows; newly ready inputs
        // can occupy free slots without waiting for those checkpoints to finish.
        while (active.size() < size_t(capacity) && !pending.empty()) {
            auto ready = std::find_if(pending.begin(), pending.end(), [&](const auto &job) {
                return job->retry_epoch < resource_epoch;
            });
            if (ready == pending.end()) break;
            auto job = *ready;
            pending.erase(ready);
            size_t input;
            if (!job->waiting_inputs.empty()) {
                input = job->waiting_inputs.front();
                job->waiting_inputs.pop_front();
            } else input = job->next_lane++;
            active.push_back({{++lane_sequence, job->execution_id, input}, job, {}});
            if (job->next_lane < job->requests.size() || !job->waiting_inputs.empty())
                pending.push_back(job);
        }
        bool new_prefill = std::any_of(active.begin(), active.end(),
                                       [](const Active &row) { return !row.prefilled; });
        if (!new_prefill && !preparation_busy)
            dispatch_preparation();
        if (active.empty() && releases.empty() && request_releases.empty())
            return;
        ExecutionBatch batch;
        batch.release.swap(releases);
        batch.release_requests.swap(request_releases);
        for (auto &entry : active) {
            if (entry.job->plan)
                batch.requests.push_back({entry.job->execution_id, std::move(entry.job->plan)});
            batch.rows.push_back(entry.row);
            entry.row.create = false;
        }
        // Prefill runs first; newly ready rows join this iteration's decode.
        // Reserve the previous decode duration from the audio frame period.
        // An overloaded engine still advances one layer to avoid starvation.
        batch.prefill_budget_ms = std::max(0., frame_period_ms - last_decode_ms);
        batch.output = std::move(output_pool.back());
        output_pool.pop_back();
        ++inflight_batches;
        executor.submit(std::move(batch));
    }
    void complete(Completion &result) {
        Range trace("tts.engine.complete");
        bool released = result.resources_released;
        if (result.executed) {
            if (!inflight_batches)
                throw std::logic_error("unexpected execution completion");
            --inflight_batches;
        }
        if (result.decoded_rows)
            last_decode_ms = result.decode_ms;
        if (stopping)
            return;
        if (result.fatal) {
            executor_error = result.fatal;
            std::vector<std::shared_ptr<Job>> interrupted;
            for (auto &row : active)
                interrupted.push_back(row.job);
            for (auto *queue : {&pending, &planning, &preparing})
                interrupted.insert(interrupted.end(), queue->begin(), queue->end());
            active.clear();
            pending.clear();
            planning.clear();
            preparing.clear();
            releases.clear();
            for (auto it = jobs.begin(); it != jobs.end();) {
                auto job = (it++)->second;
                finish(job, result.fatal);
                retire(job);
            }
            for (const auto &job : interrupted) {
                finish(job, result.fatal);
                retire(job);
            }
            for (auto &control : controls)
                control(result.fatal);
            controls.clear();
            return;
        }
        std::map<Job *, GenerationFrame> output;
        std::vector<std::shared_ptr<Job>> updated;
        for (size_t i = 0; i < result.valid_rows; ++i) {
            auto &out = result.rows[i];
            auto found = std::find_if(active.begin(), active.end(),
                                      [&](const Active &a) { return a.row.id == out.id; });
            // Cancellation can retire a row while its queued execution drains.
            if (found == active.end())
                continue;
            auto job = found->job;
            if (out.deferred) {
                job->waiting_inputs.push_back(found->row.input);
                job->retry_epoch = resource_epoch;
                active.erase(found);
                if (std::find(pending.begin(), pending.end(), job) == pending.end())
                    pending.push_back(job);
                continue;
            }
            released |= out.done || (!found->prefilled && out.prefilled);
            if (std::find(updated.begin(), updated.end(), job) == updated.end())
                updated.push_back(job);
            auto &frame = output[job.get()];
            frame.batch_size = result.decoded_rows;
            if (out.error)
                finish(job, out.error);
            job->stats.prefill_ms += out.stats.prefill_ms - found->previous.prefill_ms;
            job->stats.audio_seconds += out.stats.audio_seconds - found->previous.audio_seconds;
            job->stats.frames = std::max(job->stats.frames, out.stats.frames);
            job->stats.truncated |= out.stats.truncated;
            found->previous = out.stats;
            found->prefilled = out.prefilled;
            if (!out.pcm.empty())
                frame.chunks.emplace_back(int(found->row.input), out.pcm);
            if (out.done) {
                ++job->completed;
                active.erase(found);
            }
        }
        for (auto &job : updated) {
            publish(job, std::move(output[job.get()]));
            retire(job);
        }
        if (released) ++resource_epoch;
    }
    void execute(Command command) {
        switch (command.kind) {
        case Command::Kind::Run:
            if (command.run)
                command.run();
            break;
        case Command::Kind::Add:
            enqueue(std::move(command.job));
            break;
        case Command::Kind::Cancel: {
            auto it = jobs.find(command.request_id);
            if (it != jobs.end())
                cancel(it->second);
            break;
        }
        case Command::Kind::Shutdown:
            stop_work();
            break;
        case Command::Kind::Prepared:
            command.prepared(std::move(command.preparation), command.preparation_error);
            break;
        case Command::Kind::Planned:
            if (!stopping)
                prepared_results.push_back(std::move(command));
            break;
        }
    }
    void schedule() {
        log::Scope scope(engine_id);
        while (!stopping) {
            auto deadline = expire_reads();
            Command command;
            if (!planning.empty() || !prepared_results.empty()) {
                incoming.try_dequeue(command);
            } else if (deadline == Clock::time_point::max()) {
                incoming.wait_dequeue(command);
            } else {
                auto us =
                    std::chrono::duration_cast<std::chrono::microseconds>(deadline - Clock::now())
                        .count();
                incoming.wait_dequeue_timed(command, std::max<int64_t>(0, us));
            }
            expire_reads();
            execute(std::move(command));
            service_commands();
            if (stopping)
                break;
            dispatch();
            plan_ready();
            dispatch();
        }
        // The owner has stopped ingress; reject any remaining queued queries.
        Command discarded;
        while (incoming.try_dequeue(discarded))
            execute(std::move(discarded));
    }
    void service_commands() {
        Command command;
        // A bounded drain also makes progress under continuous client queries.
        for (size_t i = 0, limit = request_limit + active.size() + control_count + 1;
             i < limit && !stopping && incoming.try_dequeue(command); ++i)
            execute(std::move(command));
    }
    std::unique_ptr<PreparationTask> prepare_task(std::unique_ptr<PreparationTask> task) {
        bool done = false;
        std::unique_ptr<PreparationTask> result;
        std::exception_ptr error;
        executor.prepare_task(std::move(task), [this, &done, &result, &error](auto returned,
                                                                           auto failure) {
            Command command;
            command.kind = Command::Kind::Prepared;
            command.preparation = std::move(returned);
            command.preparation_error = failure;
            command.prepared = [&done, &result, &error](auto value, auto exception) {
                result = std::move(value);
                error = exception;
                done = true;
            };
            send(incoming, std::move(command));
        });
        // Continue the CPU actor's event loop while the executor owns the task.
        // The reply itself is a queue event; no polling or shared completion flag.
        while (!done) {
            Command command;
            if (planning.empty() && prepared_results.empty())
                incoming.wait_dequeue(command);
            else
                incoming.try_dequeue(command);
            execute(std::move(command));
            if (!stopping) {
                service_commands();
                expire_reads();
                dispatch();
                plan_ready();
                dispatch();
            }
        }
        if (error)
            std::rethrow_exception(error);
        return result;
    }
};
GenerationFrame RequestScheduler::read(const std::string &request_id, int timeout_ms) {
    if (timeout_ms < -1)
        throw std::invalid_argument("timeout must be -1 or nonnegative");
    Reply<GenerationFrame> reply;
    impl_->post([&, id = request_id] {
        try {
            impl_->read(id, timeout_ms, reply);
        } catch (...) {
            reply.fail(std::current_exception());
        }
    });
    return reply.get();
}
void RequestScheduler::cancel(const std::string &request_id) {
    try {
        impl_->post(Impl::Command{Impl::Command::Kind::Cancel, {}, request_id});
    } catch (...) {
        if (impl_->state == Impl::State::Stopped)
            return;
        throw;
    }
}
void RequestScheduler::run_sync(
    std::function<void()> operation,
    std::function<void(const PreparationExecutor &)> prepare) {
    Reply<void> reply;
    impl_->post([&, operation = std::move(operation), prepare = std::move(prepare)]() mutable {
        bool admitted = false;
        try {
            impl_->check_running();
            if (impl_->control_count >= impl_->request_limit)
                throw RequestOverloaded("control queue is full");
            ++impl_->control_count;
            admitted = true;
            if (prepare)
                prepare([&](auto task) { return impl_->prepare_task(std::move(task)); });
            impl_->check_running();
            auto id = ++impl_->sequence;
            impl_->controls.push_back(
                [&, id, operation = std::move(operation)](std::exception_ptr error) {
                    if (!error) try {
                        log::Scope scope(impl_->engine_id, id);
                        operation();
                    } catch (...) {
                        error = std::current_exception();
                    }
                    // Release actor-owned capacity before publishing the reply.
                    // Queued controls rejected by shutdown use this same path.
                    impl_->post([&, error = std::move(error)] {
                        impl_->busy = false;
                        --impl_->control_count;
                        if (error) reply.fail(error);
                        else reply.run([] {});
                    });
                });
        } catch (...) {
            if (admitted) --impl_->control_count;
            reply.fail(std::current_exception());
        }
    });
    reply.get();
}
void RequestScheduler::stop() {
    impl_->stop();
}
void RequestScheduler::run_async() {
    impl_->start();
}
RequestScheduler::RequestScheduler(RequestPlanner &planner, ModelRunner &runner, int batch,
                                   double frame_period_ms, uint64_t engine_id)
    : impl_(std::make_unique<Impl>(planner, runner, batch, frame_period_ms, engine_id)) {}
RequestScheduler::~RequestScheduler() = default;
void RequestScheduler::configure(size_t request_limit, size_t output_buffer_bytes) {
    if (!request_limit || !output_buffer_bytes)
        throw std::invalid_argument("serving limits must be positive");
    impl_->call([&] { impl_->configure(request_limit, output_buffer_bytes); });
}
void RequestScheduler::submit(const Request &request) {
    if (impl_->state != Impl::State::Running)
        throw impl_->unavailable();
    if (request.request_id.empty() || request.request_id.find('\0') != std::string::npos)
        throw std::invalid_argument("request_id must be nonempty and contain no NUL");
    if (request.inputs.empty())
        throw std::invalid_argument("empty request batch");
    auto job = impl_->snapshot(request);
    try {
        impl_->post(Impl::Command{Impl::Command::Kind::Add, {}, {}, std::move(job)});
    } catch (...) {
        send(impl_->available, true);
        throw;
    }
}
std::string RequestScheduler::submit_generated(const std::vector<Input> &inputs,
                                               const GenerationOptions &options) {
    if (inputs.empty())
        throw std::invalid_argument("empty request batch");
    return impl_->call([&] {
        std::string id;
        do {
            id = "offline:" + std::to_string(++impl_->sequence);
        } while (impl_->jobs.count(id));
        impl_->enqueue(impl_->snapshot({id, inputs, options}));
        return id;
    });
}
} // namespace tts
