#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <stdexcept>
namespace tts {
using Json = nlohmann::json;
inline Json read_json(const std::string &path) {
    std::ifstream f(std::filesystem::u8path(path));
    if (!f)
        throw std::runtime_error("cannot open " + path);
    return Json::parse(f);
}
} // namespace tts
