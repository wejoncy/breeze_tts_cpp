#pragma once
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>

namespace tts {
// Initialization owns plan creation; serving only looks up the sealed working
// set. Plans remain at stable addresses until backend destruction.
template <class Key, class Value> class PreparedCache {
    std::map<Key, std::unique_ptr<Value>> entries_;
    bool sealed_ = false;

  public:
    // Initialization publishes the complete working set. Serving only looks up
    // retained plans; a missing shape cannot silently tune or create resources.
    void seal() {
        sealed_ = true;
    }
    bool sealed() const {
        return sealed_;
    }
    template <class Create> Value &prepare(const Key &key, Create &&create) {
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            if (sealed_)
                throw std::logic_error("prepared resource missing after initialization");
            auto value = create();
            if (!value)
                throw std::logic_error("null prepared plan");
            it = entries_.emplace(key, std::move(value)).first;
        }
        return *it->second;
    }
    Value *find(const Key &key) const {
        auto it = entries_.find(key);
        return it == entries_.end() ? nullptr : it->second.get();
    }
    template <class Use> void prepare_all(Use &&use) {
        if (sealed_)
            throw std::logic_error("resource preparation after sealing");
        for (auto &entry : entries_)
            use(*entry.second);
    }
    size_t size() const {
        return entries_.size();
    }
};
} // namespace tts
