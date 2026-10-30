#include "engine/request_scheduler.h"
#include "engine/request_executor.h"
#include "runtime/mailbox.h"
#include "support/log.h"
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
    Stats stats;
    std::exception_ptr error;
    size_t buffered = 0, next_lane = 0, completed = 0, buffer_limit = 0;
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
        enum class Kind { Run, Add, Cancel, Shutdown };
        Kind kind = Kind::Run;
        std::function<void()> run;
        std::string request_id;
        std::unique_ptr<Job> job;
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
    Mailbox<Command> incoming;
    // Admission transfers one slot through this queue; it shares no counter.
    Mailbox<bool> available;
    // The CPU scheduler exclusively owns the mutable scheduling state below.
    std::unordered_map<std::string, std::shared_ptr<Job>> jobs;
    std::deque<std::shared_ptr<Job>> planning, pending;
    std::vector<Active> active;
    std::vector<uint64_t> releases;
    std::vector<uint64_t> request_releases;
    std::deque<std::function<void(std::exception_ptr)>> controls;
    size_t request_limit = 32, issued_slots = 32, buffer_limit = 1 << 20, control_count = 0;
    bool stopping = false, busy = false;
    std::exception_ptr executor_error;
    // The caller owns lifecycle operations and stops ingress before shutdown.
    enum class State { Ready, Running, Stopped };
    State state = State::Ready;
    RequestExecutor executor;
    std::thread scheduler;
    Impl(RequestPlanner &p, ModelRunner &r, int batch, double period, uint64_t id)
        : planner(p), capacity(batch), frame_period_ms(period), engine_id(id),
          executor(r, id, [this](Completion result) {
              send(incoming,
                   Command{Command::Kind::Run, [this, result = std::move(result)]() mutable {
                               complete(std::move(result));
                           }});
          }) {
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
        if (job->released || std::find(pending.begin(), pending.end(), job) != pending.end() ||
            std::any_of(active.begin(), active.end(),
                        [&](const Active &a) { return a.job == job; }))
            return;
        job->released = true;
        request_releases.push_back(job->execution_id);
        job->plan.reset();
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
        pending.clear();
        executor.stop();
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
            // Report the protocol error without replacing or retiring live work.
            Result rejected;
            rejected.error = std::make_exception_ptr(std::invalid_argument("duplicate request_id"));
            emit(it->second, std::move(rejected));
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
    }
    void publish(const std::shared_ptr<Job> &job, GenerationFrame frame) {
        if (job->cancelled || job->done)
            return;
        if (frame.chunks.empty() && job->completed != job->requests.size())
            return;
        Result result;
        for (const auto &chunk : frame.chunks)
            result.bytes += chunk.second.size() * sizeof(int16_t);
        if (result.bytes > job->buffer_limit - job->buffered) {
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
    void dispatch() {
        if (busy || stopping || executor_error)
            return;
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
            auto control = std::move(controls.front());
            controls.pop_front();
            busy = true;
            executor.control([this, control = std::move(control)](std::exception_ptr error) {
                control(error);
                send(incoming, Command{Command::Kind::Run, [this] {
                                           busy = false;
                                           --control_count;
                                       }});
            });
            return;
        }
        // Keep decode rows resident and pack all ready inputs into free slots.
        // A prefill batch retains its layer checkpoint until it completes.
        bool prefilling = std::any_of(active.begin(), active.end(),
                                      [](const Active &row) { return !row.prefilled; });
        while (!prefilling && active.size() < size_t(capacity) && !pending.empty()) {
            auto job = pending.front();
            pending.pop_front();
            active.push_back({{++lane_sequence, job->execution_id, job->next_lane++}, job, {}});
            if (job->next_lane < job->requests.size())
                pending.push_back(job);
        }
        if (active.empty() && releases.empty() && request_releases.empty())
            return;
        ExecutionBatch batch;
        batch.release.swap(releases);
        batch.release_requests.swap(request_releases);
        for (const auto &entry : active) {
            if (entry.job->plan)
                batch.requests.push_back({entry.job->execution_id, std::move(entry.job->plan)});
            batch.rows.push_back(entry.row);
            if (!entry.prefilled)
                batch.prefill.push_back(entry.row.id);
        }
        // Prefill runs first; newly ready rows join this iteration's decode.
        // Reserve the previous decode duration from the audio frame period.
        // An overloaded engine still advances one layer to avoid starvation.
        batch.prefill_budget_ms = std::max(0., frame_period_ms - last_decode_ms);
        busy = true;
        executor.submit(std::move(batch));
    }
    void complete(Completion result) {
        busy = false;
        if (result.decoded_rows)
            last_decode_ms = result.decode_ms;
        if (stopping)
            return;
        if (result.fatal) {
            executor_error = result.fatal;
            active.clear();
            pending.clear();
            planning.clear();
            releases.clear();
            for (auto it = jobs.begin(); it != jobs.end();) {
                auto job = (it++)->second;
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
        for (auto &out : result.rows) {
            auto found = std::find_if(active.begin(), active.end(),
                                      [&](const Active &a) { return a.row.id == out.id; });
            if (found == active.end())
                throw std::logic_error("unknown completed lane");
            auto job = found->job;
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
                frame.chunks.emplace_back(int(found->row.input), std::move(out.pcm));
            if (out.done) {
                ++job->completed;
                active.erase(found);
            }
        }
        for (auto &job : updated) {
            publish(job, std::move(output[job.get()]));
            retire(job);
        }
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
        }
    }
    void schedule() {
        log::Scope scope(engine_id);
        for (;;) {
            auto deadline = expire_reads();
            Command command;
            if (!planning.empty()) {
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
            if (stopping)
                break;
            dispatch();
            if (planning.empty())
                continue;
            auto job = planning.front();
            planning.pop_front();
            if (job->cancelled) {
                retire(job);
                continue;
            }
            try {
                job->plan = planner.prepare(job->requests, job->options);
                pending.push_back(job);
            } catch (...) {
                finish(job, std::current_exception());
                retire(job);
            }
            dispatch();
        }
        // The owner has stopped ingress; reject any remaining queued queries.
        Command discarded;
        while (incoming.try_dequeue(discarded))
            execute(std::move(discarded));
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
void RequestScheduler::run_sync(std::function<void()> operation) {
    Reply<void> reply;
    impl_->post([&, operation = std::move(operation)]() mutable {
        try {
            impl_->check_running();
            if (impl_->control_count >= impl_->request_limit)
                throw RequestOverloaded("control queue is full");
            auto id = ++impl_->sequence;
            impl_->controls.push_back(
                [&, id, operation = std::move(operation)](std::exception_ptr error) {
                    reply.run([&] {
                        if (error)
                            std::rethrow_exception(error);
                        log::Scope scope(impl_->engine_id, id);
                        operation();
                    });
                });
            ++impl_->control_count;
        } catch (...) {
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
