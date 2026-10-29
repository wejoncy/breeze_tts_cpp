#pragma once
#include "tts/tts.h"
#include <spdlog/fmt/fmt.h>
#include <string_view>

namespace tts::log {
bool enabled(tts_log_level) noexcept;
bool in_callback() noexcept;
void check_not_callback();
tts_log_level parse_level(std::string_view);
void configure(const tts_log_config &);
void emit(tts_log_level, const char *component, const std::string &message) noexcept;
uint64_t next_engine_id() noexcept;
class Scope {
    uint64_t engine_, request_;

  public:
    Scope(uint64_t engine, uint64_t request = 0) noexcept;
    ~Scope();
    Scope(const Scope &) = delete;
};
template <typename... Args>
void write(tts_log_level level, const char *component, fmt::format_string<Args...> format,
           Args &&...args) noexcept {
    try {
        emit(level, component, fmt::format(format, std::forward<Args>(args)...));
    } catch (...) { /* A diagnostic must not fail inference. */
    }
}
} // namespace tts::log
// Arguments and formatting are skipped when the level is disabled.
#define TTS_LOG(level, component, ...)                                                             \
    do {                                                                                           \
        if (::tts::log::enabled(level))                                                            \
            ::tts::log::write(level, component, __VA_ARGS__);                                      \
    } while (false)
