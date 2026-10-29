#include "safetensors.h"
#include "json.h"
#include <cmath>
#include <cstring>
namespace tts {
void SafeTensors::open_file(const std::string &path, const std::string &prefix) {
    auto f = std::make_unique<MappedFile>(path);
    uint64_t header;
    std::memcpy(&header, f->data(), 8);
    if (header > 100000000 || header > f->size() - 8)
        throw std::runtime_error("invalid safetensors header");
    const char *base = static_cast<const char *>(f->data()) + 8 + header;
    auto j = Json::parse(std::string(static_cast<const char *>(f->data()) + 8, header));
    std::unordered_map<std::string, Entry> pending;
    for (const auto &[name, v] : j.items()) {
        if (name == "__metadata__")
            continue;
        auto dtype = v.at("dtype").get<std::string>();
        Type type;
        if (dtype == "BF16")
            type = Type::BF16;
        else if (dtype == "F32")
            type = Type::F32;
        else
            throw std::runtime_error("unsupported tensor dtype: " + dtype);
        std::vector<int> shape;
        size_t count = 1;
        for (const auto &d : v.at("shape")) {
            int n = d.get<int>();
            if (n <= 0 || count > SIZE_MAX / size_t(n))
                throw std::runtime_error("invalid tensor shape");
            shape.push_back(n);
            count *= n;
        }
        const auto &offsets = v.at("data_offsets");
        if (offsets.size() != 2)
            throw std::runtime_error("invalid tensor offsets");
        double da = offsets[0].get<double>(), dz = offsets[1].get<double>();
        if (da < 0 || dz < da || dz > double(f->size() - 8 - header) || da != std::floor(da) ||
            dz != std::floor(dz))
            throw std::runtime_error("tensor outside file");
        size_t a = size_t(da), z = size_t(dz);
        if (count > SIZE_MAX / bytes(type) || z - a != count * bytes(type))
            throw std::runtime_error("tensor byte count mismatch");
        if (entries_.count(prefix + name) ||
            !pending.emplace(prefix + name, Entry{base + a, z - a, type, shape}).second)
            throw std::runtime_error("duplicate tensor: " + name);
    }
    files_.push_back(std::move(f));
    entries_.merge(pending);
}
} // namespace tts
