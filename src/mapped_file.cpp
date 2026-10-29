#include "mapped_file.h"
#include <cstdint>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <filesystem>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
namespace tts {
MappedFile::MappedFile(const std::string &path) {
#ifdef _WIN32
    HANDLE file = CreateFileW(std::filesystem::u8path(path).c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot open " + path);
    LARGE_INTEGER length;
    if (!GetFileSizeEx(file, &length) || length.QuadPart < 8 ||
        uint64_t(length.QuadPart) > SIZE_MAX) {
        CloseHandle(file);
        throw std::runtime_error("invalid mapped file size: " + path);
    }
    size_ = size_t(length.QuadPart);
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(file);
    if (mapping) {
        data_ = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(mapping);
    }
#else
    int file = open(path.c_str(), O_RDONLY);
    if (file < 0)
        throw std::runtime_error("cannot open " + path);
    struct stat st{};
    if (fstat(file, &st) || st.st_size < 8 || uint64_t(st.st_size) > SIZE_MAX) {
        close(file);
        throw std::runtime_error("invalid mapped file size: " + path);
    }
    size_ = size_t(st.st_size);
    void *mapping = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, file, 0);
    close(file);
    if (mapping != MAP_FAILED)
        data_ = mapping;
#endif
    if (!data_)
        throw std::runtime_error("cannot map " + path);
}
MappedFile::~MappedFile() {
#ifdef _WIN32
    if (data_)
        UnmapViewOfFile(data_);
#else
    if (data_)
        munmap(const_cast<void *>(data_), size_);
#endif
}
} // namespace tts
