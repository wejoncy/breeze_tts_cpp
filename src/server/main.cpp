#include "server/server.h"
#include "support/log.h"
#include "tts/engine.h"
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <iostream>
#include <limits>
#include <map>
#include <variant>
#include <type_traits>
#include <thread>

namespace {
std::atomic<bool> interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free, "signal handler requires lock-free stores");
void handle_signal(int) {
    interrupted.store(true, std::memory_order_relaxed);
}
template <typename Integer> Integer integer(const std::string &text) {
    Integer result{};
    auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::invalid_argument("invalid integer: " + text);
    return result;
}
size_t mebibytes(size_t amount) {
    if (amount > SIZE_MAX / (1 << 20))
        throw std::invalid_argument("memory budget is too large");
    return amount * (1 << 20);
}
} // namespace

int run_server(int argc, const char *const *argv) {
    try {
        tts::http::Config config;
        std::string model, quantization;
        int device = 0, max_batch = 8, context = 2048;
        size_t runtime_memory_mb = 0, memory_limit_mb = 0;
        size_t stream_buffer_mb = config.stream_buffer_bytes >> 20;
        std::string log_level = "info";
        using Argument = std::variant<std::string *, int *, size_t *>;
        const std::map<std::string, Argument> arguments{
            {"--model", &model},
            {"--served-model-name", &config.model_name},
            {"--voice-presets-file", &config.voice_presets_file},
            {"--host", &config.host},
            {"--port", &config.port},
            {"--device", &device},
            {"--max-batch", &max_batch},
            {"--context", &context},
            {"--quantization", &quantization},
            {"--memory-limit-mb", &memory_limit_mb},
            {"--runtime-memory-mb", &runtime_memory_mb},
            {"--http-threads", &config.threads},
            {"--queue-limit", &config.queue_limit},
            {"--tts-batch-max-items", &config.batch_max_items},
            {"--stream-buffer-mb", &stream_buffer_mb},
            {"--request-timeout-seconds", &config.request_timeout_seconds},
            {"--log-level", &log_level}};
        for (int index = 1; index < argc; ++index) {
            std::string option = argv[index];
            if (option == "--help") {
                std::cout << "tts-server --model DIR [options]\n";
                for (const auto &[name, field] : arguments)
                    std::visit(
                        [&](auto *value) {
                            std::cout << "  " << name << " VALUE (current: " << *value << ")\n";
                        },
                        field);
                return 0;
            }
            auto found = arguments.find(option);
            if (found == arguments.end())
                throw std::invalid_argument("unknown argument: " + option);
            if (++index >= argc)
                throw std::invalid_argument("missing value for " + option);
            std::visit(
                [&](auto *destination) {
                    using Value = std::remove_reference_t<decltype(*destination)>;
                    if constexpr (std::is_same_v<Value, std::string>)
                        *destination = argv[index];
                    else
                        *destination = integer<Value>(argv[index]);
                },
                found->second);
        }
        config.stream_buffer_bytes = mebibytes(stream_buffer_mb);
        tts_log_config logging{tts::log::parse_level(log_level), nullptr, 0};
        if (model.empty())
            throw std::invalid_argument("--model is required");
        config.validate();
        tts::log::configure(logging);
        tts::Engine engine(model, device, max_batch, context, 128,
                           tts::quantization_from_string(quantization),
                           mebibytes(runtime_memory_mb), mebibytes(memory_limit_mb));
        tts::http::Server server(engine, config);
        server.bind();
        engine.run_async();
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
        std::atomic<bool> finished{false};
        std::thread signal_monitor([&] {
            while (!finished && !interrupted)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (interrupted)
                server.stop();
        });
        std::exception_ptr failure;
        try {
            server.listen();
        } catch (...) {
            failure = std::current_exception();
            server.stop();
        }
        finished = true;
        signal_monitor.join();
        engine.shutdown();
        if (failure)
            std::rethrow_exception(failure);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "tts-server: " << error.what() << '\n';
        return 1;
    }
}
