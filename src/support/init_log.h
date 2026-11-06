#pragma once
#include <chrono>

namespace tts::log {
// Initialization only. Implemented out of line so logging changes do not rebuild kernels.
class InitPhase {
    const char *name_;
    std::chrono::steady_clock::time_point start_;
    int exceptions_;

  public:
    explicit InitPhase(const char *name);
    ~InitPhase();
    InitPhase(const InitPhase &) = delete;
    InitPhase &operator=(const InitPhase &) = delete;
    void next(const char *name);
};
} // namespace tts::log
