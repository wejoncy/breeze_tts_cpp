#include "engine/request_executor.h"
#include "execution/model_runner.h"
#include "runtime/mailbox.h"
#include "runtime/error.h"
#include "support/log.h"
#include "support/profiler.h"
#include <thread>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

namespace tts {
struct RequestExecutor::Impl {
    struct Command {
        ExecutionBatch batch;
        std::function<void(std::exception_ptr)> control;
        bool stop = false;
        std::vector<PreparedTask> preparation;
        PreparationCallback prepared;
    };
    ModelRunner &runner;
    uint64_t engine_id;
    Mailbox<Command> ready;
    std::unordered_map<uint64_t, std::unique_ptr<ModelRunner::RequestState>> states;
    struct Request {
        std::unique_ptr<RequestPlan> plan;
        std::unique_ptr<ModelRunner::RequestAdmission> admission;
    };
    std::unordered_map<uint64_t, Request> requests;
    std::function<void(Completion)> completed;
    std::function<void()> initialize;
    std::thread worker;
    std::vector<const ExecutionRow *> rows;
    std::vector<ModelRunner::RequestState *> prefill, batch_states;
    std::vector<AudioCallback> callbacks;
    void execute(ExecutionBatch &batch, Completion &result) {
        Range trace("tts.engine.execute");
        result.resources_released = !batch.release.empty() || !batch.release_requests.empty();
        for (auto id : batch.release)
            states.erase(id);
        for (auto id : batch.release_requests)
            requests.erase(id);
        for (auto &request : batch.requests)
            requests.emplace(request.id, Request{std::move(request.plan), {}});
        std::unordered_map<uint64_t, std::exception_ptr> rejected;
        std::unordered_set<uint64_t> deferred;
        rows.clear();
        if (batch.rows.size() > result.rows.size())
            throw std::logic_error("batch exceeds initialized completion capacity");
        for (const auto &row : batch.rows) {
            // A queued batch may still name a row that finished in the previous
            // execution. Only explicit admission can create a GPU state.
            if (!row.create && !states.count(row.id))
                continue;
            rows.push_back(&row);
            auto &out = result.rows[result.valid_rows++];
            out.id = row.id;
            out.stats = {};
            out.error = {};
            out.done = out.prefilled = out.deferred = false;
            out.pcm.clear();
            if (rejected.count(row.request) || deferred.count(row.request))
                continue;
            if (!states.count(row.id)) {
                try {
                    log::Scope scope(engine_id, row.request);
                    auto &request = requests.at(row.request);
                    if (!request.admission)
                        request.admission = runner.admit(*request.plan);
                    if (!request.admission) {
                        deferred.insert(row.request);
                        continue;
                    }
                    auto state = runner.prepare(*request.plan, row.input, *request.admission);
                    if (!state) {
                        out.deferred = true;
                        continue;
                    }
                    states.emplace(row.id, std::move(state));
                } catch (const DeviceError &) {
                    throw;
                } catch (...) {
                    rejected[row.request] = std::current_exception();
                }
            }
        }
        prefill.clear();
        // A model can reach a new preparation checkpoint during the previous
        // queued step. Use executor-owned progress rather than stale metadata.
        for (const auto *row : rows) {
            auto found = states.find(row->id);
            if (found != states.end() && !found->second->prefilled() &&
                !found->second->done() && !rejected.count(row->request))
                prefill.push_back(found->second.get());
        }
        if (!prefill.empty()) {
            auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::duration<double, std::milli>(batch.prefill_budget_ms);
            do {
                runner.prefill_step(prefill);
            } while (
                std::any_of(prefill.begin(), prefill.end(),
                            [](const auto *row) { return !row->prefilled() && !row->done(); }) &&
                std::chrono::steady_clock::now() < deadline);
        }
        batch_states.clear();
        callbacks.clear();
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto &row = *rows[i];
            auto &out = result.rows[i];
            if (out.deferred || deferred.count(row.request)) {
                out.deferred = true;
                continue;
            }
            auto error = rejected.find(row.request);
            if (error != rejected.end()) {
                out.error = error->second;
                out.done = true;
            }
            if (out.done || states.at(row.id)->done() || !states.at(row.id)->prefilled())
                continue;
            batch_states.push_back(states.at(row.id).get());
            callbacks.push_back([output = &result.rows[i].pcm](int, const int16_t *pcm, size_t count) {
                if (count > output->capacity() - output->size())
                    throw std::logic_error("audio step exceeds initialized output capacity");
                output->insert(output->end(), pcm, pcm + count);
                return true;
            });
        }
        auto start = std::chrono::steady_clock::now();
        if (!batch_states.empty())
            runner.step(batch_states, callbacks);
        result.decode_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        result.decoded_rows = int(batch_states.size());
        for (size_t i = 0; i < result.valid_rows; ++i) {
            auto &out = result.rows[i];
            auto found = states.find(out.id);
            if (found == states.end())
                continue;
            auto &state = *found->second;
            out.stats = state.stats();
            out.prefilled = state.prefilled();
            out.done |= state.done();
            if (state.error()) {
                out.error = state.error();
                out.done = true;
            }
            if (out.done)
                states.erase(found);
        }
    }
    void run() {
        log::Scope scope(engine_id);
        std::exception_ptr fatal;
        try { initialize(); } catch (...) { fatal = std::current_exception(); }
        for (;;) {
            Command command;
            ready.wait_dequeue(command);
            if (command.stop)
                break;
            if (!command.preparation.empty()) {
                Range trace("tts.engine.prepare_execute");
                for (size_t i = 0; i < command.preparation.size(); ++i) {
                    auto &item = command.preparation[i];
                    if (fatal)
                        item.error = fatal;
                    else try {
                        runner.prepare_task(*item.task);
                    } catch (const DeviceError &) {
                        item.error = fatal = std::current_exception();
                        states.clear();
                        requests.clear();
                        Completion result;
                        result.fatal = fatal;
                        completed(std::move(result));
                    } catch (...) {
                        item.error = std::current_exception();
                    }
                    command.prepared(i, std::move(item));
                }
                continue;
            }
            if (command.control) {
                command.control(fatal);
                continue;
            }
            Completion result = std::move(command.batch.output);
            result.valid_rows = 0;
            result.fatal = {};
            result.decode_ms = 0;
            result.decoded_rows = 0;
            result.resources_released = false;
            result.executed = true;
            try {
                if (fatal)
                    std::rethrow_exception(fatal);
                execute(command.batch, result);
            } catch (...) {
                result.fatal = fatal = std::current_exception();
                states.clear();
                requests.clear();
            }
            completed(std::move(result));
        }
        states.clear();
        requests.clear();
    }

  public:
    Impl(ModelRunner &runner_value, int capacity, uint64_t id, std::function<void(Completion)> complete,
          std::function<void()> initialize_value)
        : runner(runner_value), engine_id(id), completed(std::move(complete)),
          initialize(std::move(initialize_value)) {
        rows.reserve(capacity);
        prefill.reserve(capacity);
        batch_states.reserve(capacity);
        callbacks.reserve(capacity);
    }
    ~Impl() {
        stop();
    }
    void start() {
        worker = std::thread([this] { run(); });
    }
    void submit(ExecutionBatch batch) {
        send(ready, Command{std::move(batch), {}, false});
    }
    void prepare_task(std::unique_ptr<PreparationTask> task,
                      std::function<void(std::unique_ptr<PreparationTask>, std::exception_ptr)> reply) {
        std::vector<PreparedTask> tasks;
        tasks.push_back({std::move(task), {}});
        prepare_tasks(std::move(tasks), [reply = std::move(reply)](size_t, PreparedTask result) {
            reply(std::move(result.task), result.error);
        });
    }
    void prepare_tasks(std::vector<PreparedTask> tasks, PreparationCallback reply) {
        Command command;
        command.preparation = std::move(tasks);
        command.prepared = std::move(reply);
        send(ready, std::move(command));
    }
    void control(std::function<void(std::exception_ptr)> operation) {
        send(ready, Command{{}, std::move(operation), false});
    }
    void stop() {
        if (!worker.joinable())
            return;
        send(ready, Command{{}, {}, true});
        worker.join();
    }
};
RequestExecutor::RequestExecutor(ModelRunner &runner, int capacity, uint64_t engine_id,
                                 std::function<void(Completion)> completed,
                                 std::function<void()> initialize)
    : impl_(std::make_unique<Impl>(runner, capacity, engine_id, std::move(completed),
                                   std::move(initialize))) {}
RequestExecutor::~RequestExecutor() = default;
void RequestExecutor::start() {
    impl_->start();
}
void RequestExecutor::submit(ExecutionBatch batch) {
    impl_->submit(std::move(batch));
}
void RequestExecutor::prepare_task(
    std::unique_ptr<PreparationTask> task,
    std::function<void(std::unique_ptr<PreparationTask>, std::exception_ptr)> reply) {
    impl_->prepare_task(std::move(task), std::move(reply));
}
void RequestExecutor::prepare_tasks(std::vector<PreparedTask> tasks, PreparationCallback reply) {
    impl_->prepare_tasks(std::move(tasks), std::move(reply));
}
void RequestExecutor::control(std::function<void(std::exception_ptr)> operation) {
    impl_->control(std::move(operation));
}
void RequestExecutor::stop() {
    impl_->stop();
}
} // namespace tts
