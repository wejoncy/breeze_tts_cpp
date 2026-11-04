#pragma once
#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tts {
class Device {
    int ordinal_ = -1;

  public:
    Device() = default;
    explicit Device(int ordinal) : ordinal_(ordinal) {
        if (ordinal < -1) throw std::invalid_argument("invalid device ordinal");
    }
    explicit Device(std::string_view name) {
        if (name == "cpu") return;
        if (name == "cuda") { ordinal_ = 0; return; }
        if (name.substr(0, 5) != "cuda:") throw std::invalid_argument("invalid device");
        auto result = std::from_chars(name.data() + 5, name.data() + name.size(), ordinal_);
        if (result.ec != std::errc{} || result.ptr != name.data() + name.size() || ordinal_ < 0)
            throw std::invalid_argument("invalid CUDA device ordinal");
    }
    static Device cpu() { return Device(); }
    static Device cuda(int ordinal = 0) {
        if (ordinal < 0) throw std::invalid_argument("invalid CUDA device ordinal");
        return Device(ordinal);
    }
    int index() const { return ordinal_; }
    bool is_cpu() const { return ordinal_ < 0; }
    bool is_cuda() const { return ordinal_ >= 0; }
    std::string str() const { return is_cpu() ? "cpu" : "cuda:" + std::to_string(ordinal_); }
    bool operator==(Device other) const { return ordinal_ == other.ordinal_; }
    bool operator!=(Device other) const { return !(*this == other); }
};
} // namespace tts
