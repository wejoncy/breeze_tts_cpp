#pragma once
#include "io/mapped_file.h"
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string_view>

namespace tts::text {
// Bounds checked readers for the checkpoint's little-endian language assets.
class DataReader {
    const unsigned char *data_;
    size_t size_, position_ = 0;
  public:
    DataReader(const void *data, size_t size)
        : data_(static_cast<const unsigned char *>(data)), size_(size) {}
    size_t remaining() const { return size_ - position_; }
    template <class T> T read() {
        if (sizeof(T) > remaining()) throw std::invalid_argument("truncated language asset");
        T result;
        std::memcpy(&result, data_ + position_, sizeof(T));
        position_ += sizeof(T);
        return result;
    }
    std::string_view text(size_t size) {
        if (size > remaining()) throw std::invalid_argument("truncated language string");
        auto result = std::string_view(reinterpret_cast<const char *>(data_ + position_), size);
        position_ += size;
        return result;
    }
    DataReader slice(size_t offset) const {
        if (offset > size_) throw std::invalid_argument("language data offset");
        return {data_ + offset, size_ - offset};
    }
    void finish() const {
        if (remaining()) throw std::invalid_argument("trailing language section data");
    }
};
class LinguisticAsset {
    MappedFile file_;
    std::map<std::string, std::pair<size_t, size_t>> sections_;
  public:
    LinguisticAsset(const std::string &path, std::string_view magic) : file_(path) {
        DataReader reader(file_.data(), file_.size());
        if (reader.text(8) != magic || reader.read<uint32_t>() != 1)
            throw std::invalid_argument("unsupported language asset format");
        auto count = reader.read<uint32_t>();
        if (count > reader.remaining() / 32) throw std::invalid_argument("language section count");
        for (uint32_t i = 0; i < count; ++i) {
            auto name = reader.text(8);
            size_t end = name.find('\0');
            auto offset = reader.read<uint64_t>(), size = reader.read<uint64_t>();
            reader.read<uint64_t>();
            if (offset > file_.size() || size > file_.size() - offset ||
                !sections_.emplace(std::string(name.substr(0, end)), std::pair{offset, size}).second)
                throw std::invalid_argument("invalid language section");
        }
    }
    DataReader section(const std::string &name) const {
        auto it = sections_.find(name);
        if (it == sections_.end()) throw std::invalid_argument("missing language section: " + name);
        return {static_cast<const unsigned char *>(file_.data()) + it->second.first,
                it->second.second};
    }
};
} // namespace tts::text
