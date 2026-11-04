#pragma once
#include "ops/storage.h"
#include "core/tensor_storage.h"
#include "io/model_loader.h"
#include <memory>
#include <cstring>
#include <unordered_map>
#include <functional>

namespace tts {
struct Weight {
    Buffer storage;
    std::vector<int> shape;
    Type type;
    TensorView view() const {
        const int rows = shape.empty() ? 1 : shape.front();
        if (rows < 1 || !storage.size())
            throw std::invalid_argument("model parameter cannot be empty");
        const size_t cols = storage.size() / bytes(type) / rows;
        if (cols > size_t(INT32_MAX))
            throw std::invalid_argument("weight exceeds tensor view dimensions");
        return storage.view(rows, int(cols), type);
    }
};

class Weights {
    BufferAllocator &allocator_;
    Quantization quantization_;
    SafeTensors files_;
    const std::vector<std::string> checkpoint_files_;
    std::unordered_map<std::string, std::unique_ptr<Weight>> loaded_;
    std::vector<std::function<void()>> uploads_;
    std::vector<std::function<void()>> postprocess_;
    enum class Phase { Registering, Loading, Loaded, Sealed };
    Phase phase_ = Phase::Registering;
    void require_loading() const {
        if (phase_ == Phase::Sealed)
            throw std::logic_error("weight allocation/upload after model initialization");
    }
    void require_unique(const std::string &name) const {
        require_registration();
        if (loaded_.count(name) || files_.has(name))
            throw std::invalid_argument("duplicate derived parameter: " + name);
    }
    void require_registration() const {
        if (phase_ != Phase::Registering)
            throw std::logic_error("parameter registration after weight loading started");
    }

  public:
    // Layers first declare destination storage and load transforms. Execute
    // these once after the network is built, before any runner/graph exists.
    void load() {
        require_registration();
        phase_ = Phase::Loading;
        for (const auto &upload : uploads_) upload();
        for (const auto &process : postprocess_) process();
        ops::synchronize(allocator_.device_ordinal());
        uploads_.clear();
        postprocess_.clear();
        phase_ = Phase::Loaded;
    }
    void after_loading(std::function<void()> process) {
        require_registration();
        postprocess_.push_back(std::move(process));
    }
    void seal() {
        require_loading();
        if (phase_ != Phase::Loaded) throw std::logic_error("model weights have not been loaded");
        // load() has synchronized every upload and load-time transform.
        files_ = SafeTensors{};
        phase_ = Phase::Sealed;
    }
    BufferAllocator &allocator() const { return allocator_; }
    Quantization quantization() const { return quantization_; }
    explicit Weights(BufferAllocator &allocator, Quantization quantization,
                      ModelLoader::Checkpoint checkpoint)
        : allocator_(allocator), quantization_(quantization), files_(std::move(checkpoint.tensors)),
          checkpoint_files_(std::move(checkpoint.files)) {}
    ~Weights() {
        // A failed load may still have H2D reads from the mapped checkpoint.
        // Pool frees are asynchronous; keep the mappings alive until reads finish.
        if (phase_ != Phase::Sealed) {
            try { ops::synchronize(allocator_.device_ordinal()); } catch (...) {}
        }
    }
    TensorView get(const std::string &name) {
        auto it = loaded_.find(name);
        if (it != loaded_.end())
            return it->second->view();
        require_registration();
        const auto &e = files_.get(name);
        auto w = std::make_unique<Weight>();
        w->shape = e.shape;
        w->type = e.type;
        w->storage = Buffer(e.size, &allocator_);
        auto view = w->view();
        uploads_.push_back([this, view, source = e.data, size = e.size] {
            ops::upload_bytes(allocator_.device_ordinal(), view.data, source, size);
        });
        loaded_[name] = std::move(w);
        return view;
    }
    TensorView optional(const std::string &name) {
        auto it = loaded_.find(name);
        if (it != loaded_.end()) return it->second->view();
        require_loading();
        return files_.has(name) ? get(name) : TensorView{};
    }
    RowScaledView quantized(const std::string &name, const std::vector<std::string> &parts) {
        require_unique(name);
        require_unique(name + ".scales");
        if (parts.empty())
            throw std::invalid_argument("quantized weight has no sources");
        int rows = 0, cols = 0;
        Type source_type = files_.get(parts.front()).type;
        for (const auto &part : parts) {
            const auto &e = files_.get(part);
            bool matrix = e.shape.size() == 2 || (e.shape.size() == 3 && e.shape[2] == 1);
            if ((source_type != Type::BF16 && source_type != Type::F16) ||
                e.type != source_type || !matrix || e.shape[0] < 1 || e.shape[1] < 1 ||
                e.shape[0] > INT32_MAX - rows || (cols && cols != e.shape[1]))
                throw std::invalid_argument(
                    "quantized weights require compatible F16/BF16 matrices or pointwise projections");
            cols = e.shape[1];
            rows += e.shape[0];
        }
        auto values = std::make_unique<Weight>(), scales = std::make_unique<Weight>();
        values->shape = {rows, cols};
        values->type = quantized_type(quantization_);
        scales->shape = {rows, 1};
        scales->type = Type::F32;
        values->storage = Buffer(size_t(rows) * cols, &allocator_);
        scales->storage = Buffer(size_t(rows) * sizeof(float), &allocator_);
        RowScaledView result{values->view(), scales->view()};
        uploads_.push_back([this, parts, rows, cols, source_type, result] {
            Buffer source(size_t(rows) * cols * sizeof(uint16_t), &allocator_);
            size_t offset = 0;
            for (const auto &part : parts) {
                const auto &e = files_.get(part);
                ops::upload_bytes(allocator_.device_ordinal(), source.ptr<char>() + offset, e.data, e.size);
                offset += e.size;
            }
            ops::quantize_rows(source.view(rows, cols, source_type), result);
            ops::synchronize(allocator_.device_ordinal()); // Reuse a single matrix's 16-bit source capacity during loading.
        });
        loaded_[name] = std::move(values);
        loaded_[name + ".scales"] = std::move(scales);
        return result;
    }
    const SafeTensors::Entry &entry(const std::string &name) const {
        require_loading();
        return files_.get(name);
    }
    void validate(const std::string &name, const std::vector<int> &shape, Type type) const {
        const auto &value = entry(name);
        if (value.shape != shape || value.type != type)
            throw std::invalid_argument("checkpoint parameter shape/type mismatch: " + name);
    }
    TensorView fused(const std::string &name, const std::vector<std::string> &parts) {
        require_unique(name);
        if (parts.empty())
            throw std::invalid_argument("fused weight has no sources");
        auto w = std::make_unique<Weight>();
        size_t size = 0;
        int rows = 0, cols = 0;
        for (const auto &p : parts) {
            const auto &e = files_.get(p);
            if (e.shape.empty() || e.shape[0] < 1 || !e.size ||
                e.shape[0] > INT32_MAX - rows || e.size > SIZE_MAX - size)
                throw std::runtime_error("invalid fused weight source: " + p);
            size_t trailing = e.size / bytes(e.type) / e.shape[0];
            if (trailing > size_t(INT32_MAX))
                throw std::invalid_argument("fused weight row exceeds tensor dimensions");
            int part_cols = int(trailing);
            if (!cols) {
                cols = part_cols;
                w->type = e.type;
            }
            if (part_cols != cols || e.type != w->type)
                throw std::runtime_error("fused weight shape");
            size += e.size;
            rows += e.shape[0];
        }
        w->shape = {rows, cols};
        w->storage = Buffer(size, &allocator_);
        auto view = w->view();
        uploads_.push_back([this, parts, view] {
            size_t pos = 0;
            for (const auto &p : parts) {
                const auto &e = files_.get(p);
                ops::upload_bytes(allocator_.device_ordinal(), static_cast<char *>(view.data) + pos, e.data, e.size);
                pos += e.size;
            }
        });
        loaded_[name] = std::move(w);
        return view;
    }
    TensorView custom(const std::string &name, const void *data, int rows, int cols, Type type) {
        require_unique(name);
        if (rows < 1 || cols < 1 || !data)
            throw std::invalid_argument("invalid custom weight");
        auto w = std::make_unique<Weight>();
        w->shape = {rows, cols};
        w->type = type;
        w->storage = Buffer(size_t(rows) * cols * bytes(type), &allocator_);
        auto v = w->view();
        std::vector<char> packed(w->storage.size());
        std::memcpy(packed.data(), data, packed.size());
        uploads_.push_back([this, v, packed = std::move(packed)] {
            ops::upload_bytes(allocator_.device_ordinal(), v.data, packed.data(), packed.size());
        });
        loaded_[name] = std::move(w);
        return v;
    }
    TensorView generated(const std::string &name, const std::vector<float> &values,
                         int rows, int cols, Type type);
    TensorView matrix_columns(const std::string &name, const std::string &source,
                              int first, int count) {
        require_loading();
        const auto &e = entry(source);
        if (e.shape.size() != 2 || first < 0 || count < 1 || first > e.shape[1] - count)
            throw std::invalid_argument("weight column partition");
        std::vector<char> packed(size_t(e.shape[0]) * count * bytes(e.type));
        for (int row = 0; row < e.shape[0]; ++row)
            std::memcpy(packed.data() + size_t(row) * count * bytes(e.type),
                static_cast<const char *>(e.data) + (size_t(row) * e.shape[1] + first) * bytes(e.type),
                size_t(count) * bytes(e.type));
        return custom(name, packed.data(), e.shape[0], count, e.type);
    }
    std::vector<float> host_values(const std::string &name) const;
    size_t size() const {
        size_t n = 0;
        for (const auto &[k, w] : loaded_)
            n += w->storage.size();
        return n;
    }
    const std::vector<std::string> &checkpoint_files() const { return checkpoint_files_; }
};
} // namespace tts
