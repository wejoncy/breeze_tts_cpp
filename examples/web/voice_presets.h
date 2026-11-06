#pragma once
#include "io/json.h"
#include <filesystem>
#include <mutex>

namespace tts::http {
class VoicePresets {
    std::filesystem::path path_;
    std::mutex mutex_;
    Json read() const;
    void write(const Json &) const;

  public:
    explicit VoicePresets(const std::string &file);
    bool enabled() const { return !path_.empty(); }
    Json list();
    Json save(const Json &);
    bool erase(const std::string &id);
};
} // namespace tts::http
