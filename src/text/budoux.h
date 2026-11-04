#pragma once
#include "io/json.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>
namespace tts::text {
class Budoux {
    struct Impl;
    std::unique_ptr<Impl> impl_;
  public:
    explicit Budoux(const Json &weights);
    ~Budoux();
    std::vector<std::u32string> split(std::u32string_view) const;
};
} // namespace tts::text
