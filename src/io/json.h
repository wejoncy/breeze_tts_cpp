#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>
namespace tts {
using Json = nlohmann::json;
// DOM callbacks make nlohmann scan parent objects after each nested object.
// Check duplicate keys with its public SAX API, then use its normal DOM parser;
// both passes stay linear for large dictionaries and safetensors indexes.
class UniqueJsonKeys final : public Json::json_sax_t {
    std::vector<std::unordered_set<std::string>> objects_;
  public:
    bool null() override { return true; }
    bool boolean(bool) override { return true; }
    bool number_integer(Json::number_integer_t) override { return true; }
    bool number_unsigned(Json::number_unsigned_t) override { return true; }
    bool number_float(Json::number_float_t, const Json::string_t &) override { return true; }
    bool string(Json::string_t &) override { return true; }
    bool binary(Json::binary_t &) override { return true; }
    bool start_object(size_t) override { objects_.emplace_back(); return true; }
    bool end_object() override { objects_.pop_back(); return true; }
    bool start_array(size_t) override { return true; }
    bool end_array() override { return true; }
    bool key(Json::string_t &name) override {
        if (!objects_.back().insert(name).second)
            throw std::invalid_argument("duplicate JSON key: " + name);
        return true;
    }
    bool parse_error(size_t, const std::string &, const Json::exception &error) override {
        throw std::invalid_argument(error.what());
    }
};
inline Json parse_json(std::string_view text) {
    UniqueJsonKeys keys;
    Json::sax_parse(text, &keys);
    return Json::parse(text);
}
inline Json read_json(const std::string &path) {
    std::ifstream f(std::filesystem::u8path(path));
    if (!f)
        throw std::runtime_error("cannot open " + path);
    UniqueJsonKeys keys;
    Json::sax_parse(f, &keys);
    f.clear();
    f.seekg(0);
    return Json::parse(f);
}
} // namespace tts
