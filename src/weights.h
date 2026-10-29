#pragma once
#include "backend.h"
#include "tensor_storage.h"
#include "safetensors.h"
#include <memory>
#include <cstring>
#include <unordered_map>

namespace tts {
struct Weight {
    Buffer storage;
    std::vector<int> shape;
    Type type;
    TensorView view() const {
        return storage.view(shape[0], int(storage.size() / bytes(type) / shape[0]), type);
    }
};

class Weights {
    Backend &gpu_;
    SafeTensors files_;
    std::unordered_map<std::string, std::unique_ptr<Weight>> loaded_;
    bool sealed_ = false;
    void require_loading() const {
        if (sealed_)
            throw std::logic_error("weight allocation/upload after model initialization");
    }

  public:
    void seal() {
        sealed_ = true;
    }
    Backend &backend() {
        return gpu_;
    }
    explicit Weights(Backend &b, SafeTensors files) : gpu_(b), files_(std::move(files)) {}
    ~Weights() {
        // A failed load may still have H2D reads from the mapped checkpoint.
        // Pool frees are asynchronous; keep the mappings alive until reads finish.
        cudaStreamSynchronize(gpu_.stream);
    }
    TensorView get(const std::string &name) {
        auto it = loaded_.find(name);
        if (it != loaded_.end())
            return it->second->view();
        require_loading();
        const auto &e = files_.get(name);
        auto w = std::make_unique<Weight>();
        w->shape = e.shape;
        w->type = e.type;
        w->storage = Buffer(e.size, &gpu_.memory);
        gpu_.upload(w->storage.data(), e.data, e.size);
        auto view = w->view();
        loaded_[name] = std::move(w);
        return view;
    }
    TensorView optional(const std::string &name) {
        return files_.has(name) ? get(name) : TensorView{};
    }
    RowScaledView quantized(const std::string &name, const std::vector<std::string> &parts) {
        if (loaded_.count(name))
            return {loaded_.at(name)->view(), loaded_.at(name + ".scales")->view()};
        require_loading();
        int rows = 0, cols = 0;
        for (const auto &part : parts) {
            const auto &e = files_.get(part);
            if (e.type != Type::BF16 || e.shape.size() != 2 || (cols && cols != e.shape[1]))
                throw std::invalid_argument("quantized weight requires compatible BF16 matrices");
            cols = e.shape[1];
            rows += e.shape[0];
        }
        auto values = std::make_unique<Weight>(), scales = std::make_unique<Weight>();
        values->shape = {rows, cols};
        values->type = gpu_.quantized().type();
        scales->shape = {rows, 1};
        scales->type = Type::F32;
        values->storage = Buffer(size_t(rows) * cols, &gpu_.memory);
        scales->storage = Buffer(size_t(rows) * sizeof(float), &gpu_.memory);
        Buffer source(size_t(rows) * cols * sizeof(uint16_t), &gpu_.memory);
        size_t offset = 0;
        for (const auto &part : parts) {
            const auto &e = files_.get(part);
            gpu_.upload(source.ptr<char>() + offset, e.data, e.size);
            offset += e.size;
        }
        RowScaledView result{values->view(), scales->view()};
        gpu_.quantized().quantize(source.view(rows, cols), result);
        gpu_.sync(); // Explicit load-time conversion; the BF16 source is then released.
        loaded_[name] = std::move(values);
        loaded_[name + ".scales"] = std::move(scales);
        return result;
    }
    const SafeTensors::Entry &entry(const std::string &name) const {
        return files_.get(name);
    }
    TensorView fused(const std::string &name, const std::vector<std::string> &parts) {
        if (loaded_.count(name))
            return loaded_.at(name)->view();
        require_loading();
        auto w = std::make_unique<Weight>();
        size_t size = 0;
        int rows = 0, cols = 0;
        for (const auto &p : parts) {
            const auto &e = files_.get(p);
            if (!cols) {
                cols = e.shape[1];
                w->type = e.type;
            }
            if (e.shape.size() != 2 || e.shape[1] != cols || e.type != w->type)
                throw std::runtime_error("fused weight shape");
            size += e.size;
            rows += e.shape[0];
        }
        w->shape = {rows, cols};
        w->storage = Buffer(size, &gpu_.memory);
        size_t pos = 0;
        for (const auto &p : parts) {
            const auto &e = files_.get(p);
            gpu_.upload(static_cast<char *>(w->storage.data()) + pos, e.data, e.size);
            pos += e.size;
        }
        auto view = w->view();
        loaded_[name] = std::move(w);
        return view;
    }
    TensorView custom(const std::string &name, const void *data, int rows, int cols, Type type) {
        if (loaded_.count(name))
            return loaded_.at(name)->view();
        require_loading();
        auto w = std::make_unique<Weight>();
        w->shape = {rows, cols};
        w->type = type;
        w->storage = Buffer(size_t(rows) * cols * bytes(type), &gpu_.memory);
        gpu_.upload(w->storage.data(), data, w->storage.size());
        gpu_.sync();
        auto v = w->view();
        loaded_[name] = std::move(w);
        return v;
    }
    size_t size() const {
        size_t n = 0;
        for (const auto &[k, w] : loaded_)
            n += w->storage.size();
        return n;
    }
};
} // namespace tts
