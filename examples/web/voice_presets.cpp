#include "voice_presets.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <random>
#include <set>
#include <sstream>
#include <system_error>
#ifdef _WIN32
#include <windows.h>
#endif

namespace tts::http {
namespace {
std::string new_id() {
    std::random_device random;
    std::uniform_int_distribution<unsigned> byte(0, 255);
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) result << std::setw(2) << byte(random);
    return result.str();
}
bool valid_id(const std::string &id) {
    return id.size() == 32 && std::all_of(id.begin(), id.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}
std::string seed_text(const Json &value) {
    if (value.is_number_unsigned()) return std::to_string(value.get<uint64_t>());
    if (value.is_number_integer() && value.get<int64_t>() >= 0)
        return std::to_string(value.get<int64_t>());
    if (value.is_string()) {
        const auto text = value.get<std::string>();
        uint64_t seed;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), seed);
        if (parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size())
            return std::to_string(seed);
    }
    throw std::invalid_argument("seed must be an unsigned 64-bit integer");
}
Json checked_preset(const Json &value) {
    if (!value.is_object()) throw std::invalid_argument("voice preset must be an object");
    for (const auto &[name, field] : value.items())
        if (name != "id" && name != "name" && name != "instructions" && name != "cfg_scale" && name != "seed" &&
            name != "input" && name != "ref_text")
            throw std::invalid_argument("unsupported voice preset field: " + name);
    for (const auto *name : {"name", "instructions"})
        if (!value.contains(name) || !value[name].is_string() || value[name].get_ref<const std::string &>().find('\0') != std::string::npos)
            throw std::invalid_argument(std::string("invalid ") + name);
    if (value["name"].get_ref<const std::string &>().find_first_not_of(" \t\r\n") == std::string::npos)
        throw std::invalid_argument("voice preset name is required");
    if (!value.contains("cfg_scale") || !value["cfg_scale"].is_number() ||
        !std::isfinite(value["cfg_scale"].get<double>()) || value["cfg_scale"].get<double>() < 0)
        throw std::invalid_argument("invalid cfg_scale");
    if (!value.contains("seed")) throw std::invalid_argument("seed is required");
    const auto id = value.value("id", std::string{});
    if (!id.empty() && !valid_id(id)) throw std::invalid_argument("invalid voice preset id");
    Json entry{{"id", id}, {"name", value["name"]}, {"instructions", value["instructions"]},
               {"cfg_scale", value["cfg_scale"]}, {"seed", seed_text(value["seed"])}};
    // Older presets have no text fields; keep their absence distinct from deliberately empty text.
    for (const auto *name : {"input", "ref_text"}) {
        if (!value.contains(name)) continue;
        if (!value[name].is_string() || value[name].get_ref<const std::string &>().find('\0') != std::string::npos)
            throw std::invalid_argument(std::string("invalid ") + name);
        entry[name] = value[name];
    }
    return entry;
}
} // namespace

VoicePresets::VoicePresets(const std::string &file) : path_(std::filesystem::u8path(file)) {}

Json VoicePresets::read() const {
    if (!enabled() || !std::filesystem::exists(path_)) return {{"version", 1}, {"presets", Json::array()}};
    try {
        auto data = read_json(path_.u8string());
        if (!data.is_object() || data.value("version", 0) != 1 || !data.contains("presets") || !data["presets"].is_array())
            throw std::runtime_error("invalid voice preset file structure");
        std::set<std::string> ids;
        for (auto &entry : data["presets"]) {
            entry = checked_preset(entry);
            const auto id = entry["id"].get<std::string>();
            if (id.empty() || !ids.insert(id).second) throw std::runtime_error("missing or duplicate voice preset id");
        }
        return data;
    } catch (const std::exception &error) {
        throw std::runtime_error(std::string("cannot read voice preset file: ") + error.what());
    }
}

void VoicePresets::write(const Json &data) const {
    if (!enabled()) throw std::runtime_error("voice preset storage is not configured");
    if (!path_.parent_path().empty()) std::filesystem::create_directories(path_.parent_path());
    auto temporary = path_;
    temporary += "." + new_id() + ".tmp";
    try {
        {
            std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
            if (!file) throw std::runtime_error("cannot create voice preset file");
            file << data.dump(2) << '\n';
            file.close();
            if (!file) throw std::runtime_error("cannot write voice preset file");
        }
        // Replace a complete file in the same directory; failed writes keep the old JSON.
#ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), path_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::system_error(GetLastError(), std::system_category(), "cannot replace voice preset file");
#else
        std::filesystem::rename(temporary, path_);
#endif
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

Json VoicePresets::list() {
    std::lock_guard<std::mutex> lock(mutex_);
    return read()["presets"];
}

Json VoicePresets::save(const Json &value) {
    auto entry = checked_preset(value);
    std::lock_guard<std::mutex> lock(mutex_);
    auto data = read();
    auto &presets = data["presets"];
    const auto id = entry["id"].get<std::string>();
    if (id.empty()) {
        do { entry["id"] = new_id(); }
        while (std::any_of(presets.begin(), presets.end(), [&](const Json &other) { return other["id"] == entry["id"]; }));
    } else {
        const auto previous = std::find_if(presets.begin(), presets.end(), [&](const Json &other) { return other["id"] == id; });
        if (previous == presets.end()) throw std::out_of_range("voice preset not found");
        for (const auto *name : {"input", "ref_text"})
            if (!entry.contains(name) && previous->contains(name)) entry[name] = (*previous)[name];
        presets.erase(previous);
    }
    presets.insert(presets.begin(), entry);
    write(data);
    return entry;
}

bool VoicePresets::erase(const std::string &id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto data = read();
    auto &presets = data["presets"];
    const auto entry = std::find_if(presets.begin(), presets.end(), [&](const Json &other) { return other["id"] == id; });
    if (entry == presets.end()) return false;
    presets.erase(entry);
    write(data);
    return true;
}
} // namespace tts::http
