#include "server/server.h"
#include "support/log.h"
#include "tts/engine.h"
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <iostream>
#include <limits>
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
size_t mebibytes(const std::string &text) {
    auto amount = integer<size_t>(text);
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
        size_t runtime_memory = 0, memory_limit = 0;
        tts_log_config logging{TTS_LOG_INFO, nullptr, 0, nullptr, nullptr};
        for (int index = 1; index < argc; ++index) {
            std::string option = argv[index];
            auto value = [&]() -> std::string {
                if (++index >= argc)
                    throw std::invalid_argument("missing value for " + option);
                return argv[index];
            };
            if (option == "--help") {
                std::cout
                    << "tts-server --model DIR [--host 127.0.0.1] [--port 8088]\n"
                       "  [--served-model-name NAME] [--device 0] [--max-batch 8] [--context "
                       "2048]\n"
                       "  [--quantization fp8|int8] [--tts-batch-max-items 32]\n"
                       "  [--memory-limit-mb 0] [--runtime-memory-mb 0]\n"
                       "  [--http-threads 64] [--queue-limit 32] [--stream-buffer-mb 1]\n"
                       "  [--request-timeout-seconds 300] [--log-level info]\n"
                       "POST /v1/audio/speech: {input:..., response_format:wav|pcm, stream:false}\n"
                       "  stream_format=sse enables SSE; streaming requires response_format=pcm\n"
                       "POST /v1/audio/speech/batch: {items:[{input:...},...]} -> JSON audio "
                       "results\n"
                       "GET /health, /v1/models, /v1/model, /v1/audio/speech/{speech_id}\n";
                return 0;
            } else if (option == "--model")
                model = value();
            else if (option == "--served-model-name")
                config.model_name = value();
            else if (option == "--host")
                config.host = value();
            else if (option == "--port")
                config.port = integer<int>(value());
            else if (option == "--device")
                device = integer<int>(value());
            else if (option == "--max-batch")
                max_batch = integer<int>(value());
            else if (option == "--context")
                context = integer<int>(value());
            else if (option == "--quantization")
                quantization = value();
            else if (option == "--memory-limit-mb")
                memory_limit = mebibytes(value());
            else if (option == "--runtime-memory-mb")
                runtime_memory = mebibytes(value());
            else if (option == "--http-threads")
                config.threads = integer<size_t>(value());
            else if (option == "--queue-limit")
                config.queue_limit = integer<size_t>(value());
            else if (option == "--tts-batch-max-items")
                config.batch_max_items = integer<size_t>(value());
            else if (option == "--stream-buffer-mb")
                config.stream_buffer_bytes = mebibytes(value());
            else if (option == "--request-timeout-seconds")
                config.request_timeout_seconds = integer<int>(value());
            else if (option == "--log-level")
                logging.level = tts::log::parse_level(value());
            else
                throw std::invalid_argument("unknown argument: " + option);
        }
        if (model.empty())
            throw std::invalid_argument("--model is required");
        config.validate();
        tts::log::configure(logging);
        tts::Engine engine(model, device, max_batch, context, 128,
                           tts::quantization_from_string(quantization), runtime_memory,
                           memory_limit);
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
