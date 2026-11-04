#include "io/safetensors.h"
#include "io/json.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <set>
namespace tts {
namespace {
uint64_t nonnegative_integer(const Json &value) {
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<int64_t>() < 0))
        throw std::runtime_error("safetensors dimensions/offsets must be nonnegative integers");
    return value.get<uint64_t>();
}
} // namespace
std::vector<std::string> SafeTensors::open_checkpoint(const std::string &directory,
                                                     const std::string &prefix) {
    const std::string index_name = "model.safetensors.index.json";
    if (!std::filesystem::exists(std::filesystem::u8path(directory + "/" + index_name))) {
        std::set<std::string> files;
        for (const auto &file : std::filesystem::directory_iterator(std::filesystem::u8path(directory)))
            if (file.is_regular_file() && file.path().extension() == ".safetensors")
                files.insert(file.path().filename().u8string());
        if (files.empty())
            throw std::runtime_error("no safetensors checkpoint in " + directory);
        for (const auto &file : files) open_file(directory + "/" + file, prefix);
        return {files.begin(), files.end()};
    }
    const auto index = read_json(directory + "/" + index_name);
    const auto &map = index.at("weight_map");
    if (!map.is_object() || map.empty())
        throw std::invalid_argument("safetensors index has no weights");
    std::set<std::string> shards;
    for (const auto &[name, value] : map.items()) {
        const auto path = value.get<std::string>();
        if (name.empty() || path.empty() || path.find_first_of("/\\:") != std::string::npos ||
            path.find("..") != std::string::npos)
            throw std::invalid_argument("invalid safetensors shard path");
        shards.insert(path);
    }
    std::vector<std::string> paths{index_name};
    for (const auto &path : shards) {
        for (const auto &name : open_file(directory + "/" + path, prefix)) {
            auto expected = map.find(name);
            if (expected == map.end() || expected->get<std::string>() != path)
                throw std::runtime_error("safetensors index shard mismatch: " + name);
        }
        paths.push_back(path);
    }
    for (const auto &[name, value] : map.items()) get(prefix + name);
    return paths;
}
std::vector<std::string> SafeTensors::open_file(const std::string &path,
                                              const std::string &prefix) {
    auto f = std::make_unique<MappedFile>(path);
    uint64_t header;
    std::memcpy(&header, f->data(), 8);
    if (header > 100000000 || header > f->size() - 8)
        throw std::runtime_error("invalid safetensors header");
    const char *base = static_cast<const char *>(f->data()) + 8 + header;
    auto j = parse_json(std::string_view(static_cast<const char *>(f->data()) + 8, header));
    if (!j.is_object())
        throw std::runtime_error("safetensors header must be an object");
    std::unordered_map<std::string, Entry> pending;
    std::vector<std::string> names;
    std::vector<std::pair<size_t, size_t>> ranges;
    for (const auto &[name, v] : j.items()) {
        if (name == "__metadata__")
            continue;
        auto dtype = v.at("dtype").get<std::string>();
        Type type;
        if (dtype == "BF16")
            type = Type::BF16;
        else if (dtype == "F16")
            type = Type::F16;
        else if (dtype == "F32")
            type = Type::F32;
        else
            throw std::runtime_error("unsupported tensor dtype: " + dtype);
        std::vector<int> shape;
        size_t count = 1;
        if (!v.at("shape").is_array())
            throw std::runtime_error("safetensors shape must be an array");
        for (const auto &d : v.at("shape")) {
            uint64_t n = nonnegative_integer(d);
            if (n > INT32_MAX || (n && count > SIZE_MAX / size_t(n)))
                throw std::runtime_error("invalid tensor shape");
            shape.push_back(int(n));
            count *= n;
        }
        const auto &offsets = v.at("data_offsets");
        if (!offsets.is_array() || offsets.size() != 2)
            throw std::runtime_error("invalid tensor offsets");
        uint64_t da = nonnegative_integer(offsets[0]), dz = nonnegative_integer(offsets[1]);
        if (dz < da || dz > f->size() - 8 - header)
            throw std::runtime_error("tensor outside file");
        size_t a = size_t(da), z = size_t(dz);
        if (count > SIZE_MAX / bytes(type) || z - a != count * bytes(type))
            throw std::runtime_error("tensor byte count mismatch");
        if (entries_.count(prefix + name) ||
            !pending.emplace(prefix + name, Entry{base + a, z - a, type, shape}).second)
            throw std::runtime_error("duplicate tensor: " + name);
        names.push_back(name);
        ranges.emplace_back(a, z);
    }
    std::sort(ranges.begin(), ranges.end());
    size_t end = 0;
    for (const auto &[first, last] : ranges) {
        if (first != end)
            throw std::runtime_error("safetensors data overlap or gap");
        end = last;
    }
    if (end != f->size() - 8 - header)
        throw std::runtime_error("safetensors unreferenced trailing data");
    files_.push_back(std::move(f));
    entries_.merge(pending);
    return names;
}
} // namespace tts
