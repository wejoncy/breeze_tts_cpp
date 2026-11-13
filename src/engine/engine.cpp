#include "tts/engine.h"
#include "engine/request_scheduler.h"
#include "engine/control_request.h"
#include "execution/model_runtime.h"
#include "execution/output_sampler.h"
#include "runtime/error.h"
#include "runtime/inference_metrics.h"
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <optional>
#include <thread>
#include "support/log.h"
#include <map>
namespace tts {
constexpr double bytes_per_mib = 1024.0 * 1024.0;
struct Engine::Impl {
    std::unique_ptr<ModelRuntime> model;
    ModelInputs inputs;
    int max_batch;
    std::unique_ptr<RequestScheduler> scheduler;
    std::unique_ptr<OutputSampler> sampler;
    std::thread worker;
    enum class State { Ready, Running, Stopped };
    std::atomic<State> state{State::Ready};
    using Clock = std::chrono::steady_clock;
    InferenceMetrics metrics;
    struct Job {
        std::string request_id;
        std::vector<Input> requests;
        GenerationOptions options;
        uint64_t execution_id = 0;
        std::unique_ptr<RequestPlan> plan;
        std::unique_ptr<RequestPreparation> preparation;
        std::unique_ptr<PreparationTask> task;
        Clock::time_point submitted;
        std::exception_ptr error;
        size_t completed = 0;
        bool done = false, cancelled = false, released = false, truncated = false;
        bool aborted() const { return cancelled || error; }
        Stats stats() const {
            return {std::chrono::duration<double, std::milli>(Clock::now() - submitted).count(),
                    int(truncated)};
        }
    };
    // Registry and preparation continuations are Engine work. Only Scheduler
    // owns whether an input is waiting or belongs to a running generation batch.
    std::unordered_map<std::string, std::shared_ptr<Job>> jobs;
    std::unordered_map<uint64_t, std::shared_ptr<Job>> execution_jobs;
    std::vector<std::shared_ptr<Job>> preparing_jobs;
    std::deque<std::shared_ptr<Job>> cpu_preparations, gpu_preparations;
    uint64_t execution_sequence = 0;
    bool stopping = false, generation_due = false;
    struct Command {
        enum class Kind { None, Add, Cancel, Control, Shutdown };
        Kind kind = Kind::None;
        Request request;
        Clock::time_point submitted = Clock::now();
        std::string request_id;
        std::shared_ptr<ControlRequest> control;
    };
    // The only cross-thread queues: request ingress and shared audio output.
    std::mutex input_mutex;
    std::condition_variable input_available;
    std::deque<Command> incoming;
    Engine::OutputQueue outgoing;
    size_t request_limit = 32, admitted_requests = 0;
    std::atomic<uint64_t> request_sequence{0};
    std::deque<std::shared_ptr<ControlRequest>> controls;
    std::shared_ptr<ControlRequest> active_control;
    std::unique_ptr<ReferencePreparation> reference_preparation;
    std::unique_ptr<ReferencePlan> reference_plan;
    std::shared_ptr<ModelRunner::AudioDecodeCache> audio_decode;
    std::exception_ptr execution_error;
    std::vector<Completion> output_pool;
    Impl(const std::string &dir, const ModelOptions &options)
        : model(load_model(dir, options)), inputs(model->take_inputs()), max_batch(options.max_batch),
          metrics(model->info().audio.sample_rate) {
        scheduler = std::make_unique<RequestScheduler>(
            model->memory(), model->kv_cache(), model->cache_slots(), max_batch);
        sampler = std::make_unique<OutputSampler>(model->memory(), model->runner().max_readback_bytes());
        output_pool.resize(2);
        for (auto &output : output_pool) {
            output.rows.resize(size_t(max_batch));
            for (auto &row : output.rows) {
                row.pcm.resize(model->runner().max_step_samples());
                row.pcm.clear();
            }
        }
    }
    ~Impl() { stop(); }
    void start() {
        if (state == State::Running) return;
        if (state == State::Stopped) throw std::runtime_error("engine is shut down");
        state = State::Running;
        try { worker = std::thread([this] { run(); }); }
        catch (...) { finish_shutdown(); state = State::Stopped; throw; }
    }
    void stop() {
        if (state == State::Stopped) return;
        if (state == State::Running) {
            Command command;
            command.kind = Command::Kind::Shutdown;
            post(std::move(command));
            worker.join();
        }
        state = State::Stopped;
        {
            std::lock_guard<std::mutex> lock(outgoing.mutex);
            outgoing.closed = true;
        }
        outgoing.available.notify_all();
    }
    std::runtime_error unavailable() const {
        return std::runtime_error(state == State::Stopped ? "engine is shut down"
            : "engine is not running; call run_async() first");
    }
    void post(Command command, bool reserve = false) {
        {
            std::lock_guard<std::mutex> lock(input_mutex);
            if (state != State::Running) throw unavailable();
            if (reserve && admitted_requests >= request_limit)
                throw RequestOverloaded("request queue is full");
            incoming.push_back(std::move(command));
            if (reserve) ++admitted_requests;
        }
        input_available.notify_all();
    }
    void release_slot() {
        std::lock_guard<std::mutex> lock(input_mutex);
        --admitted_requests;
    }
    void configure(size_t limit, size_t output_bytes) {
        {
            std::lock_guard<std::mutex> lock(input_mutex);
            request_limit = limit;
        }
        std::lock_guard<std::mutex> lock(outgoing.mutex);
        outgoing.limit = output_bytes;
    }
    void check_running() const {
        if (stopping) throw std::runtime_error("engine is shutting down");
        if (execution_error) std::rethrow_exception(execution_error);
    }
    void retire(const std::shared_ptr<Job> &job) {
        if (job->released || std::find(preparing_jobs.begin(), preparing_jobs.end(), job) != preparing_jobs.end() ||
            std::find(cpu_preparations.begin(), cpu_preparations.end(), job) != cpu_preparations.end() ||
            std::find(gpu_preparations.begin(), gpu_preparations.end(), job) != gpu_preparations.end() ||
            scheduler->contains(job->execution_id))
            return;
        job->released = true;
        metrics.retire(job->execution_id);
        model->runner().release_if_idle(job->execution_id);
        execution_jobs.erase(job->execution_id);
        job->plan.reset();
        job->preparation.reset();
        job->task.reset();
        auto found = jobs.find(job->request_id);
        if (found != jobs.end() && found->second == job) jobs.erase(found);
        release_slot();
    }
    void stop_work() {
        stopping = true;
        for (auto it = jobs.begin(); it != jobs.end();) {
            auto job = (it++)->second;
            cancel(job);
        }
        jobs.clear();
        cpu_preparations.clear();
        gpu_preparations.clear();
        scheduler->clear();
        preparing_jobs.clear();

        std::vector<std::shared_ptr<Job>> interrupted;
        for (const auto &entry : execution_jobs) interrupted.push_back(entry.second);
        for (const auto &job : interrupted) retire(job);
    }
    void add(Request request, Clock::time_point submitted) {
        auto job = std::make_shared<Job>();
        job->request_id = std::move(request.request_id);
        job->requests = std::move(request.inputs);
        job->options = request.options;
        job->submitted = submitted;
        auto [it, inserted] = jobs.emplace(job->request_id, job);
        if (!inserted) {
            release_slot();
            // A duplicate must not publish a terminal record for the live ID.
            TTS_LOG(TTS_LOG_WARN, "request", "ignored duplicate request");
            return;
        }
        job->execution_id = ++execution_sequence;
        execution_jobs.emplace(job->execution_id, job);
        try {
            check_running();
            scheduler->add(job->execution_id, job->requests.size());
            cpu_preparations.push_back(job);
        } catch (...) {
            finish(job, std::current_exception());
            retire(job);
        }
    }
    void emit(const std::shared_ptr<Job> &job, GenerationFrame output) {
        output.request_id = job->request_id;
        size_t bytes = 0;
        for (const auto &chunk : output.chunks) bytes += chunk.second.size() * sizeof(int16_t);
        {
            std::unique_lock<std::mutex> lock(outgoing.mutex);
            if (bytes > outgoing.limit - std::min(outgoing.bytes, outgoing.limit)) {
                lock.unlock();
                fail(job, std::make_exception_ptr(std::runtime_error("audio output queue exceeded its buffer limit")));
                return;
            }
            outgoing.frames.push_back(std::move(output));
            outgoing.bytes += bytes;
        }
        outgoing.available.notify_all();
    }
    void finish(const std::shared_ptr<Job> &job, std::exception_ptr failure = {}) {
        if (job->done)
            return;
        job->error = failure;
        scheduler->remove(job->execution_id);
        job->done = true;
        GenerationFrame result;
        result.error = failure;
        result.stats = job->stats();
        result.done = true;
        result.cancelled = job->cancelled;
        retire(job);
        emit(job, std::move(result));
    }
    void cancel(std::shared_ptr<Job> job) {
        if (job->done) return;
        job->cancelled = job->done = true;
        GenerationFrame result;
        result.done = result.cancelled = true;
        result.stats = job->stats();
        scheduler->remove(job->execution_id);
        for (auto *queue : {&cpu_preparations, &gpu_preparations})
            queue->erase(std::remove(queue->begin(), queue->end(), job), queue->end());
        retire(job);
        emit(job, std::move(result));
    }
    void publish(const std::shared_ptr<Job> &job, GenerationFrame frame) {
        if (job->cancelled || job->done)
            return;
        if (frame.chunks.empty() && job->completed != job->requests.size())
            return;
        frame.done = job->completed == job->requests.size();
        frame.stats = job->stats();
        job->done = frame.done;
        metrics.record_output(job->execution_id, frame);
        if (frame.done) retire(job);
        emit(job, std::move(frame));
    }
    void continue_preparation(const std::shared_ptr<Job> &job) {
        job->task = job->preparation->next();
        if (job->task) {
            gpu_preparations.push_back(job);
        } else if (!job->preparation->ready()) {
            cpu_preparations.push_back(job);
        } else {
            job->plan = job->preparation->finish();
            job->preparation.reset();
            scheduler->ready(job->execution_id, model->runner().requirements(*job->plan));
        }
    }
    void prepare_next() {
        // Yield to queue events after each CPU plan, keeping the runner fed
        // while other requests wait for GPU conditioning.
        if (!cpu_preparations.empty()) {
            auto job = cpu_preparations.front();
            cpu_preparations.pop_front();
            if (job->aborted()) { retire(job); return; }
            try {
                if (!job->preparation) job->preparation = std::visit([&](const auto &frontend) {
                    return frontend->begin(job->requests, job->options);
                }, inputs);
                if (job->task) job->preparation->accept(std::move(job->task));
                continue_preparation(job);
            } catch (...) {
                finish(job, std::current_exception());
                retire(job);
            }
        }
    }
    PreparationBatch schedule_preparation(bool after_control) {
        PreparationBatch batch;
        if (stopping || execution_error || !preparing_jobs.empty() || (generation_due && !after_control))
            return batch;
        const size_t available_rows = scheduler->available_rows();
        while (!gpu_preparations.empty() && batch.tasks.size() < available_rows) {
            auto job = gpu_preparations.front();
            gpu_preparations.pop_front();
            if (job->aborted()) { retire(job); continue; }
            batch.tasks.emplace_back(std::move(job->task));
            preparing_jobs.push_back(job);
        }
        generation_due = !batch.tasks.empty() && scheduler->available_rows() < size_t(max_batch);
        return batch;
    }
    void fail(const std::shared_ptr<Job> &job, std::exception_ptr error) {
        if (job->cancelled) return;
        job->done = false;
        finish(job, std::move(error));
        retire(job);
    }
    void cancel(const std::string &request_id) {
        auto found = jobs.find(request_id);
        if (found != jobs.end()) cancel(found->second);
    }
    void finish_shutdown() { if (!stopping) stop_work(); }
    bool has_cpu_work() const { return !cpu_preparations.empty(); }
    void commit_preparation(PreparationBatch batch) {
        if (batch.tasks.size() != preparing_jobs.size()) throw std::logic_error("preparation ownership mismatch");
        for (size_t index = 0; index < batch.tasks.size(); ++index) {
            auto job = std::move(preparing_jobs[index]);
            auto &task = batch.tasks[index];
            if (job->aborted() || stopping) { retire(job); continue; }
            try {
                if (task.error) std::rethrow_exception(task.error);
                task.task->accept(std::move(task.results));
                job->task = std::move(task.task);
                cpu_preparations.push_back(job);
            } catch (...) { finish(job, std::current_exception()); retire(job); }
        }
        preparing_jobs.clear();
    }
    ExecutionBatch schedule() {
        if (stopping || execution_error) return {};
        auto selected = scheduler->schedule();
        ExecutionBatch batch;
        batch.rows = std::move(selected.rows);
        batch.release = std::move(selected.release);
        batch.release_requests = std::move(selected.release_requests);
        for (auto id : selected.requests) {
            auto &job = execution_jobs.at(id);
            batch.requests.push_back({id, std::move(job->plan)});
        }
        for (auto &[id, error] : selected.rejected) {
            auto job = execution_jobs.at(id);
            finish(job, std::move(error));
            retire(job);
        }
        return batch;
    }
    void complete(Completion &result) {
        if (stopping) return;
        if (result.fatal) {
            scheduler->clear();
            cpu_preparations.clear(); gpu_preparations.clear();
            preparing_jobs.clear();
            std::vector<std::shared_ptr<Job>> interrupted;
            for (const auto &entry : execution_jobs) interrupted.push_back(entry.second);
            for (const auto &job : interrupted) { finish(job, result.fatal); retire(job); }
            return;
        }
        std::map<Job *, GenerationFrame> frames;
        std::vector<std::shared_ptr<Job>> updated;
        for (size_t i = 0; i < result.valid_rows; ++i) {
            const auto &out = result.rows[i];
            const auto *scheduled = scheduler->row(out.id);
            if (!scheduled) continue; // Cancelled while this batch was draining.
            const auto row = *scheduled;
            auto job = execution_jobs.at(row.request);
            if (std::find(updated.begin(), updated.end(), job) == updated.end()) updated.push_back(job);
            auto &frame = frames[job.get()];
            frame.batch_size = result.decoded_rows;
            job->truncated |= out.truncated;
            if (!out.pcm.empty()) frame.chunks.emplace_back(int(row.input), out.pcm);
            if (out.done) ++job->completed;
            scheduler->update(out.id, out.done, out.prefilled);
            if (out.error) finish(job, out.error);
        }
        for (auto &job : updated) {
            publish(job, std::move(frames[job.get()]));
            retire(job);
        }
        if (result.executed) metrics.record(result.stats);
    }
    void log_stats() {
        const auto now = Clock::now();
        const auto cache = model->runner().cache_usage();
        if (!metrics.should_log(cache, now)) return;
        const auto [running, waiting] = scheduler->request_counts();
        metrics.log({running, waiting}, cache, now);
    }
    void finish_control(const std::shared_ptr<ControlRequest> &request,
                        ControlResult result = {}, std::exception_ptr error = {}) {
        {
            std::lock_guard<std::mutex> lock(input_mutex);
            request->result = std::move(result);
            request->error = std::move(error);
            request->complete = true;
        }
        input_available.notify_all();
    }
    void execute(Command command) {
        switch (command.kind) {
        case Command::Kind::None: break;
        case Command::Kind::Add:
            if (stopping) release_slot();
            else add(std::move(command.request), command.submitted);
            break;
        case Command::Kind::Cancel: cancel(command.request_id); break;
        case Command::Kind::Control:
            if (stopping || execution_error || controls.size() + size_t(bool(active_control)) >= request_limit)
                finish_control(command.control, {}, execution_error ? execution_error : std::make_exception_ptr(
                    std::runtime_error(stopping ? "engine is shutting down" : "control queue is full")));
            else controls.push_back(std::move(command.control));
            break;
        case Command::Kind::Shutdown:
            reject_controls(std::make_exception_ptr(std::runtime_error("engine is shutting down")));
            finish_shutdown();
            break;
        }
    }
    bool take_command(Command &command) {
        std::lock_guard<std::mutex> lock(input_mutex);
        if (incoming.empty()) return false;
        command = std::move(incoming.front());
        incoming.pop_front();
        return true;
    }
    void service_requests(bool block_if_idle = false) {
        if (block_if_idle && !has_cpu_work()) {
            std::unique_lock<std::mutex> lock(input_mutex);
            input_available.wait(lock, [&] { return !incoming.empty(); });
        }
        Command command;
        for (size_t i = 0, limit = request_limit + size_t(max_batch) + controls.size() + 1;
             i < limit && !stopping && take_command(command); ++i) execute(std::move(command));
    }
    void reject_controls(std::exception_ptr error) {
        if (active_control) {
            finish_control(active_control, {}, error);
            active_control.reset();
            reference_preparation.reset();
            reference_plan.reset();
            audio_decode.reset();
        }
        for (const auto &request : controls) finish_control(request, {}, error);
        controls.clear();
    }
    bool boundary_work_ready() const {
        return !generation_due && (active_control || !controls.empty() ||
            (!gpu_preparations.empty() && preparing_jobs.empty() &&
             scheduler->available_rows()));
    }
    void commit(Completion &output) {
        if (output.fatal) { execution_error = output.fatal; reject_controls(output.fatal); }
        complete(output);
    }
    void attach_output(ExecutionBatch &batch) {
        if (output_pool.empty()) throw std::logic_error("execution pipeline output storage exhausted");
        batch.output = std::move(output_pool.back());
        output_pool.pop_back();
    }
    ForwardOutputs forward(ExecutionBatch batch) {
        generation_due = false;
        auto &runner = model->runner();
        if (!runner.has_submitted()) {
            // Cancellation can arrive after selection. Never start a new cache
            // for a row that Scheduler has already removed. Submitted frames
            // retain their exact membership until their result is collected.
            batch.rows.erase(std::remove_if(batch.rows.begin(), batch.rows.end(),
                [&](const ExecutionRow &row) { return !scheduler->row(row.id); }), batch.rows.end());
            batch.requests.erase(std::remove_if(batch.requests.begin(), batch.requests.end(),
                [&](const ExecutionRequest &request) { return !scheduler->contains(request.id); }), batch.requests.end());
        }
        return runner.forward(std::move(batch));
    }
    void complete_active_control(ControlResult result = {}, std::exception_ptr error = {}) {
        auto request = std::move(active_control);
        reference_preparation.reset();
        reference_plan.reset();
        audio_decode.reset();
        generation_due = scheduler->available_rows() < size_t(max_batch);
        finish_control(request, std::move(result), std::move(error));
    }
    void sample(ForwardOutputs outputs) {
        if (!model->runner().finished()) sampler->sample(outputs);
    }
    void prepare_batch(std::optional<ExecutionBatch> &next) {
        if (next) return;
        auto batch = schedule();
        if (!batch.empty()) { attach_output(batch); next = std::move(batch); }
    }
    void forward_next(std::optional<ExecutionBatch> &next) {
        auto batch = std::move(*next);
        next.reset();
        sample(forward(std::move(batch)));
    }
    // Called at a completed GPU program/frame boundary or to launch the next
    // control quantum. Cache ownership survives ordinary generation in between.
    void control() {
        auto &runner = model->runner();
        try {
            if (runner.preparing()) {
                auto batch = runner.collect_preparation();
                auto &output = batch.tasks.front();
                if (output.error) std::rethrow_exception(output.error);
                output.task->accept(std::move(output.results));
                reference_preparation->accept(std::move(output.task));
                generation_due = scheduler->available_rows() < size_t(max_batch);
                return;
            }
            if (runner.has_control()) {
                if (active_control->kind == ControlRequest::Kind::Decode) {
                    auto audio = runner.collect_codes();
                    if (!audio.empty()) complete_active_control(std::move(audio));
                    else generation_due = scheduler->available_rows() < size_t(max_batch);
                } else {
                    auto outputs = runner.collect_evaluation();
                    if (active_control->kind == ControlRequest::Kind::Encode) {
                        auto codes = outputs.at("codes").view();
                        complete_active_control(std::vector<int>(codes.ptr<int>(), codes.ptr<int>() + codes.numel()));
                    } else {
                        auto result = std::visit([&](const auto &frontend) {
                            return frontend->finish_reference(*reference_plan, std::move(outputs));
                        }, inputs);
                        complete_active_control(std::move(result));
                    }
                }
                return;
            }
            if (!active_control) {
                active_control = std::move(controls.front());
                controls.pop_front();
                if (active_control->kind == ControlRequest::Kind::Reference)
                    reference_preparation = std::visit([&](const auto &frontend) {
                        return frontend->begin_reference(active_control->mono,
                            std::move(active_control->transcript), active_control->language);
                    }, inputs);
                else if (active_control->kind == ControlRequest::Kind::Decode)
                    audio_decode = runner.prepare_codes(active_control->codes);
            }
            switch (active_control->kind) {
            case ControlRequest::Kind::Configure:
                configure(active_control->request_limit, active_control->output_bytes);
                complete_active_control();
                break;
            case ControlRequest::Kind::Encode:
                sample(runner.forward_encode(active_control->mono));
                break;
            case ControlRequest::Kind::Decode:
                sample(runner.forward_codes(audio_decode));
                break;
            case ControlRequest::Kind::Reference:
                if (auto task = reference_preparation->next()) {
                    PreparationBatch batch;
                    batch.tasks.emplace_back(std::move(task));
                    sample(runner.forward(std::move(batch)));
                } else {
                    reference_plan = reference_preparation->finish();
                    reference_preparation.reset();
                    sample(runner.forward_reference(*reference_plan));
                }
                break;
            }
        } catch (...) { fail_control(std::current_exception()); }
    }
    void fail_control(std::exception_ptr error) {
        try { std::rethrow_exception(error); }
        catch (const DeviceError &) {
            complete_active_control({}, error);
            throw;
        } catch (...) {
            // Failure cleanup alone drains work submitted before an exception.
            // Normal control progress shares the main event-query/commit path.
            try { model->drain(); }
            catch (...) { complete_active_control({}, std::current_exception()); throw; }
            sampler->discard_after_drain();
            model->runner().discard_control();
            complete_active_control({}, error);
        }
    }
    // One host thread drives scheduling and GPU launch. CUDA performs the
    // computation concurrently while prepare_next() builds the queued successor.
    void run() {
        auto &runner = model->runner();
        std::optional<ExecutionBatch> next;
        try { model->initialize_execution(); }
        catch (...) { execution_error = std::current_exception(); }
        while (!stopping) {
            try {
                service_requests();
                if (stopping) break;
                if (execution_error) {
                    Completion failure;
                    failure.fatal = execution_error;
                    commit(failure);
                    service_requests(true);
                    continue;
                }
                if (sampler->pending()) {
                    prepare_next();
                    if (runner.can_submit_next()) {
                        prepare_batch(next);
                        // Submit successor compute before querying the preceding
                        // result. A control quantum runs at a generation boundary.
                        if (!runner.has_submitted() && next && !boundary_work_ready())
                            runner.submit_next(*next);
                    }
                    if (!sampler->ready()) {
                        if (!has_cpu_work()) std::this_thread::yield();
                        continue;
                    }
                    runner.commit(sampler->collect());
                    sample(runner.forward());
                    if (!runner.finished()) continue;
                }
                if (runner.has_batch()) {
                    auto result = runner.collect();
                    // A submitted successor already overlaps this CPU work.
                    // Publish ready audio before launching/adopting another batch.
                    commit(result);
                    output_pool.push_back(std::move(result));
                    if (next && (runner.has_submitted() || !boundary_work_ready()))
                        forward_next(next);
                    log_stats();
                    continue;
                }
                const bool control_boundary = runner.has_control() || (runner.preparing() && preparing_jobs.empty());
                if (runner.preparing()) {
                    if (preparing_jobs.empty()) control();
                    else commit_preparation(runner.collect_preparation());
                } else if (runner.has_control()) control();

                prepare_batch(next);
                if (!control_boundary && (active_control || !controls.empty()) && (!generation_due || !next)) {
                    control();
                    continue;
                }
                auto preparation = schedule_preparation(control_boundary);
                if (!preparation.tasks.empty()) {
                    sample(runner.forward(std::move(preparation)));
                    continue;
                }
                if (next) {
                    forward_next(next);
                    continue;
                }
                if (active_control || !controls.empty()) {
                    control();
                    continue;
                }
                const bool had_cpu_work = has_cpu_work();
                prepare_next();
                if (!had_cpu_work && !has_cpu_work()) service_requests(true);
            } catch (...) {
                auto error = std::current_exception();
                if (active_control && (runner.has_control() || (runner.preparing() && preparing_jobs.empty()))) {
                    try { fail_control(error); continue; }
                    catch (...) { error = std::current_exception(); }
                }
                execution_error = error;
                try { runner.abort(); } catch (...) {}
                sampler->discard_after_drain();
                next.reset();
            }
        }
        try { runner.abort(); } catch (...) {}
        sampler->discard_after_drain();
        finish_shutdown();
        state = State::Stopped;
        Command command;
        while (take_command(command)) execute(std::move(command));
        {
            std::lock_guard<std::mutex> lock(outgoing.mutex);
            outgoing.closed = true;
        }
        outgoing.available.notify_all();
        input_available.notify_all();
    }
    ControlResult call(ControlRequest request) {
        auto operation = std::make_shared<ControlRequest>(std::move(request));
        Command command;
        command.kind = Command::Kind::Control;
        command.control = operation;
        post(std::move(command));
        std::unique_lock<std::mutex> lock(input_mutex);
        input_available.wait(lock, [&] { return operation->complete || state == State::Stopped; });
        if (!operation->complete) throw unavailable();
        if (operation->error) std::rethrow_exception(operation->error);
        return std::move(operation->result);
    }
};
void Engine::run_async() {
    impl_->start();
}
void Engine::add_request(const Request &request) {
    if (impl_->state != Impl::State::Running) throw impl_->unavailable();
    if (request.request_id.empty() || request.request_id.find('\0') != std::string::npos)
        throw std::invalid_argument("request_id must be nonempty and contain no NUL");
    if (request.inputs.empty()) throw std::invalid_argument("empty request batch");
    Impl::Command command;
    command.kind = Impl::Command::Kind::Add;
    command.request = request; // Own the snapshot before publishing to ingress.
    impl_->post(std::move(command), true);
}
Engine::OutputQueue &Engine::output_queue() { return impl_->outgoing; }
void Engine::cancel_request(const std::string &request_id) {
    if (impl_->state == Impl::State::Stopped) return;
    Impl::Command command;
    command.kind = Impl::Command::Kind::Cancel;
    command.request_id = request_id;
    impl_->post(std::move(command));
}
void Engine::shutdown() {
    impl_->stop();
}
void Engine::configure_serving(size_t request_limit, size_t output_buffer_bytes) {
    if (!request_limit || !output_buffer_bytes) throw std::invalid_argument("serving limits must be positive");
    ControlRequest request;
    request.kind = ControlRequest::Kind::Configure;
    request.request_limit = request_limit;
    request.output_bytes = output_buffer_bytes;
    impl_->call(std::move(request));
}
Engine::Engine(const std::string &dir, int device, int batch, int context, int page,
               Quantization quantization, size_t runtime_memory_bytes, size_t memory_limit_bytes) {
    auto start = std::chrono::steady_clock::now();
    TTS_LOG(TTS_LOG_INFO, "engine",
            "loading device={} batch_capacity={} context={} quantization={}", device, batch,
            context, quantization_name(quantization));
    impl_ = std::make_unique<Impl>(dir, ModelOptions{device, batch, context, page, quantization,
                                                   runtime_memory_bytes, memory_limit_bytes});
    if (log::enabled(TTS_LOG_INFO)) {
        auto ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        TTS_LOG(TTS_LOG_INFO, "engine",
                "ready elapsed_ms={:.3f} weights={:.0f} MiB", ms, weight_bytes() / bytes_per_mib);
    }
}
Engine::~Engine() = default;
size_t Engine::weight_bytes() const {
    return impl_->model->info().weight_bytes;
}
AudioFormat Engine::audio_format() const {
    return impl_->model->info().audio;
}
const ModelInfo &Engine::model_info() const {
    return impl_->model->info();
}
GenerationOptions Engine::default_options() const {
    return impl_->model->default_options();
}
int Engine::batch_capacity() const {
    return impl_->max_batch;
}
size_t Engine::max_reference_samples() const {
    return impl_->model->info().max_reference_samples;
}
std::string Engine::begin_generation(const std::vector<Input> &requests, const GenerationOptions &options) {
    if (requests.empty()) throw std::invalid_argument("empty request batch");
    Request request{"offline:" + std::to_string(++impl_->request_sequence), requests, options};
    auto id = request.request_id;
    add_request(request);
    return id;
}
std::shared_ptr<const Reference> Engine::prepare_reference(const std::vector<float> &mono,
                                                         std::string transcript,
                                                         const std::string &language) {
    ControlRequest request;
    request.kind = ControlRequest::Kind::Reference;
    request.mono = mono;
    request.transcript = std::move(transcript);
    request.language = language;
    return std::get<std::shared_ptr<const Reference>>(impl_->call(std::move(request)));
}
std::vector<int> Engine::encode_audio(const std::vector<float> &mono) {
    if (impl_->model->info().audio.codebooks <= 0)
        throw std::invalid_argument("loaded model does not expose discrete audio encoding");
    ControlRequest request;
    request.kind = ControlRequest::Kind::Encode;
    request.mono = mono;
    return std::get<std::vector<int>>(impl_->call(std::move(request)));
}
std::vector<float> Engine::decode_codes(const std::vector<int> &codes) {
    if (impl_->model->info().audio.codebooks <= 0)
        throw std::invalid_argument("loaded model does not expose discrete audio decoding");
    ControlRequest request;
    request.kind = ControlRequest::Kind::Decode;
    request.codes = codes;
    return std::get<std::vector<float>>(impl_->call(std::move(request)));
}
} // namespace tts
