#pragma once
#include "tensor.h"
#include "mapped_file.h"
#include <memory>
#include <unordered_map>
#include <vector>
namespace tts {
class SafeTensors {
  public:
    struct Entry {
        const void *data;
        size_t size;
        Type type;
        std::vector<int> shape;
    };

  private:
    std::vector<std::unique_ptr<MappedFile>> files_;
    std::unordered_map<std::string, Entry> entries_;

  public:
    void open_file(const std::string &path, const std::string &prefix = "");
    const Entry &get(const std::string &name) const {
        auto it = entries_.find(name);
        if (it == entries_.end())
            throw std::runtime_error("missing weight " + name);
        return it->second;
    }
    bool has(const std::string &name) const {
        return entries_.count(name);
    }
};
} // namespace tts
