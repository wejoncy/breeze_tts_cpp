#include "support/log.h"
#include "io/json.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <atomic>
#include <cstring>
#include <mutex>
#include <filesystem>

namespace tts::log {
namespace {
std::atomic<int> threshold{TTS_LOG_INFO};
std::atomic<uint64_t> engine_sequence{0};
thread_local uint64_t engine_id = 0, request_id = 0;
thread_local bool delivering = false;
constexpr const char *names[] = {"trace", "debug", "info", "warn", "error", "off"};
struct State {
    std::mutex mutex;
    std::shared_ptr<spdlog::logger> output;
    tts_log_callback callback = nullptr;
    void *user = nullptr;
    bool json = false;
};
State &state() {
    static State s;
    return s;
}
std::shared_ptr<spdlog::logger> output(const char *path, bool json) {
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
    return logger;
}
} // namespace
bool in_callback() noexcept {
    return delivering;
}
void check_not_callback() {
    if (delivering)
        throw std::logic_error("cannot reenter engine/log configuration from a log callback");
}
bool enabled(tts_log_level level) noexcept {
    return !delivering && level >= threshold.load(std::memory_order_relaxed) && level < TTS_LOG_OFF;
}
tts_log_level parse_level(std::string_view name) {
    for (int i = TTS_LOG_TRACE; i <= TTS_LOG_OFF; ++i)
        if (name == names[i])
            return static_cast<tts_log_level>(i);
    throw std::invalid_argument("log level must be trace, debug, info, warn, error or off");
}
void configure(const tts_log_config &config) {
    check_not_callback();
    if (config.level < TTS_LOG_TRACE || config.level > TTS_LOG_OFF ||
        (config.json != 0 && config.json != 1) || (config.callback && config.file_path) ||
        (config.file_path && !config.file_path[0]))
        throw std::invalid_argument("invalid logging configuration");
    auto next = config.callback ? nullptr : output(config.file_path, bool(config.json));
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.output)
        s.output->flush();
    s.output = std::move(next);
    s.callback = config.callback;
    s.user = config.user;
    s.json = bool(config.json);
    threshold.store(config.level, std::memory_order_relaxed);
}
uint64_t next_engine_id() noexcept {
    return ++engine_sequence;
}
Scope::Scope(uint64_t engine, uint64_t request) noexcept
    : engine_(engine_id), request_(request_id) {
    engine_id = engine;
    request_id = request;
}
Scope::~Scope() {
    engine_id = engine_;
    request_id = request_;
}
void emit(tts_log_level level, const char *component, const std::string &message) noexcept {
    if (!enabled(level))
        return;
    try {
        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        if (!enabled(level))
            return;
        auto now = spdlog::log_clock::now();
        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
        tts_log_record record{level, us, engine_id, request_id, component, message.c_str()};
        if (s.callback) {
            struct Guard {
                Guard() {
                    delivering = true;
                }
                ~Guard() {
                    delivering = false;
                }
            } guard;
            s.callback(s.user, &record);
        } else {
            if (!s.output)
                s.output = output(nullptr, false);
            auto text = s.json ? Json{{"timestamp_us", us},       {"level", names[level]},
                                      {"component", component},   {"engine_id", engine_id},
                                      {"request_id", request_id}, {"message", message}}
                                     .dump(-1, ' ', false, Json::error_handler_t::replace)
                               : fmt::format("[{}] engine={} request={} {}", component, engine_id,
                                             request_id, message);
            s.output->log(now, spdlog::source_loc{}, static_cast<spdlog::level::level_enum>(level),
                          text);
        }
    } catch (...) { /* Includes sink/callback failures; inference keeps its original result. */
    }
}
} // namespace tts::log
extern "C" int tts_configure_logging(const tts_log_config *config, char *error, size_t capacity) {
    try {
        tts::log::configure(config ? *config
                                   : tts_log_config{TTS_LOG_INFO, nullptr, 0, nullptr, nullptr});
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
