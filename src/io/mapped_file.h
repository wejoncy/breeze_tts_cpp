#pragma once
#include <cstddef>
#include <string>
namespace tts {
class MappedFile {
    const void *data_ = nullptr;
    size_t size_ = 0;

  public:
    explicit MappedFile(const std::string &path);
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;
    const void *data() const {
        return data_;
    }
    size_t size() const {
        return size_;
    }
};
} // namespace tts
