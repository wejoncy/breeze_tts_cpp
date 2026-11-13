#include "support/log.h"
#include "support/init_log.h"
#include "io/json.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <exception>

namespace tts::log {
InitPhase::InitPhase(const char *name)
    : name_(name), start_(std::chrono::steady_clock::now()),
      exceptions_(std::uncaught_exceptions()) {
    TTS_LOG(TTS_LOG_DEBUG, "engine.init", "begin phase={}", name_);
}
InitPhase::~InitPhase() {
    const bool failed = std::uncaught_exceptions() > exceptions_;
    TTS_LOG(failed ? TTS_LOG_ERROR : TTS_LOG_DEBUG, "engine.init", "{} phase={} elapsed_ms={:.3f}",
            failed ? "failed" : "done", name_,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_)
                .count());
}
void InitPhase::next(const char *name) {
    TTS_LOG(TTS_LOG_DEBUG, "engine.init", "done phase={} elapsed_ms={:.3f}", name_,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_)
                .count());
    name_ = name;
    start_ = std::chrono::steady_clock::now();
    TTS_LOG(TTS_LOG_DEBUG, "engine.init", "begin phase={}", name_);
}
namespace {
std::atomic<int> threshold{TTS_LOG_INFO};
constexpr const char *names[] = {"trace", "debug", "info", "warn", "error", "off"};
struct Output {
    std::shared_ptr<spdlog::logger> logger;
    bool json;
};
std::shared_ptr<const Output> output(const char *path, bool json) {
    spdlog::sink_ptr sink;
    if (path)
        sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            std::filesystem::u8path(path).native(), 10 << 20, 3);
    else
        sink = std::make_shared<spdlog::sinks::stderr_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("tts", std::move(sink));
    logger->set_level(spdlog::level::trace);
    logger->set_pattern(json ? "%v" : "%Y-%m-%dT%H:%M:%S.%e%z [%l] %v");
    logger->flush_on(spdlog::level::err);
    return std::make_shared<const Output>(Output{std::move(logger), json});
}
std::shared_ptr<const Output> &state() {
    static auto current = output(nullptr, false);
    return current;
}
} // namespace
bool enabled(tts_log_level level) noexcept {
    return level >= threshold.load(std::memory_order_relaxed) && level < TTS_LOG_OFF;
}
tts_log_level parse_level(std::string_view name) {
    for (int i = TTS_LOG_TRACE; i <= TTS_LOG_OFF; ++i)
        if (name == names[i])
            return static_cast<tts_log_level>(i);
    throw std::invalid_argument("log level must be trace, debug, info, warn, error or off");
}
void configure(const tts_log_config &config) {
    if (config.level < TTS_LOG_TRACE || config.level > TTS_LOG_OFF ||
        (config.json != 0 && config.json != 1) ||
        (config.file_path && !config.file_path[0]))
        throw std::invalid_argument("invalid logging configuration");
    std::atomic_store(&state(), output(config.file_path, bool(config.json)));
    threshold.store(config.level, std::memory_order_relaxed);
}
void emit(tts_log_level level, const char *component, const std::string &message) {
    const auto sink = std::atomic_load(&state());
    auto now = spdlog::log_clock::now();
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    std::string text;
    if (sink->json) {
        Json fields{{"timestamp_us", us}, {"level", names[level]},
                    {"component", component}, {"message", message}};
        text = fields.dump(-1, ' ', false, Json::error_handler_t::replace);
    } else {
        text = fmt::format("[{}] {}", component, message);
    }
    sink->logger->log(now, spdlog::source_loc{}, static_cast<spdlog::level::level_enum>(level), text);
}
} // namespace tts::log
extern "C" int tts_configure_logging(const tts_log_config *config, char *error, size_t capacity) {
    try {
        tts::log::configure(config ? *config
                                   : tts_log_config{TTS_LOG_INFO, nullptr, 0});
        return 0;
    } catch (const std::exception &e) {
        if (error && capacity) {
            std::strncpy(error, e.what(), capacity - 1);
            error[capacity - 1] = 0;
        }
        return -1;
    } catch (...) {
        if (error && capacity) {
            std::strncpy(error, "logging configuration failed", capacity - 1);
            error[capacity - 1] = 0;
        }
        return -1;
    }
}
