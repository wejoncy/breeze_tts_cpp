#include "engine/request_executor.h"
#include "execution/model_runner.h"
#include "runtime/mailbox.h"
#include "support/log.h"
#include <thread>
#include <chrono>
#include <unordered_map>

namespace tts {
struct RequestExecutor::Impl {
    struct Command {
        ExecutionBatch batch;
        std::function<void(std::exception_ptr)> control;
        bool stop = false;
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
    std::thread worker;
    void execute(ExecutionBatch &batch, Completion &result) {
        for (auto id : batch.release)
            states.erase(id);
        for (auto id : batch.release_requests)
            requests.erase(id);
        for (auto &request : batch.requests)
            requests.emplace(request.id, Request{std::move(request.plan), {}});
        result.rows.resize(batch.rows.size());
        std::unordered_map<uint64_t, std::exception_ptr> rejected;
        for (size_t i = 0; i < batch.rows.size(); ++i) {
            auto &row = batch.rows[i];
            auto &out = result.rows[i];
            out.id = row.id;
            if (rejected.count(row.request))
                continue;
            if (!states.count(row.id)) {
                try {
                    log::Scope scope(engine_id, row.request);
                    auto &request = requests.at(row.request);
                    if (!request.admission)
                        request.admission = runner.admit(*request.plan);
                    states.emplace(row.id,
                                   runner.prepare(*request.plan, row.input, *request.admission));
                } catch (const DeviceError &) {
                    throw;
                } catch (...) {
                    rejected[row.request] = std::current_exception();
                }
            }
        }
        std::vector<ModelRunner::RequestState *> prefill;
        for (auto id : batch.prefill) {
            auto row = std::find_if(batch.rows.begin(), batch.rows.end(),
                                    [&](const auto &row) { return row.id == id; });
            auto found = states.find(id);
            if (found != states.end() && row != batch.rows.end() && !rejected.count(row->request))
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
        std::vector<ModelRunner::RequestState *> batch_states;
        std::vector<AudioCallback> callbacks;
        for (size_t i = 0; i < batch.rows.size(); ++i) {
            auto &row = batch.rows[i];
            auto &out = result.rows[i];
            auto error = rejected.find(row.request);
            if (error != rejected.end()) {
                out.error = error->second;
                out.done = true;
            }
            if (out.done || !states.at(row.id)->prefilled())
                continue;
            batch_states.push_back(states.at(row.id).get());
            callbacks.push_back([&, i](int, const int16_t *pcm, size_t count) {
                result.rows[i].pcm.assign(pcm, pcm + count);
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
        for (auto &out : result.rows) {
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
        for (;;) {
            Command command;
            ready.wait_dequeue(command);
            if (command.stop)
                break;
            if (command.control) {
                command.control(fatal);
                continue;
            }
            Completion result;
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
    Impl(ModelRunner &r, uint64_t id, std::function<void(Completion)> complete)
        : runner(r), engine_id(id), completed(std::move(complete)) {}
    ~Impl() {
        stop();
    }
    void start() {
        worker = std::thread([this] { run(); });
    }
    void submit(ExecutionBatch batch) {
        send(ready, Command{std::move(batch), {}, false});
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
RequestExecutor::RequestExecutor(ModelRunner &runner, uint64_t engine_id,
                                 std::function<void(Completion)> completed)
    : impl_(std::make_unique<Impl>(runner, engine_id, std::move(completed))) {}
RequestExecutor::~RequestExecutor() = default;
void RequestExecutor::start() {
    impl_->start();
}
void RequestExecutor::submit(ExecutionBatch batch) {
    impl_->submit(std::move(batch));
}
void RequestExecutor::control(std::function<void(std::exception_ptr)> operation) {
    impl_->control(std::move(operation));
}
void RequestExecutor::stop() {
    impl_->stop();
}
} // namespace tts
