#pragma once
#include "ops/storage.h"
#include "core/tensor_storage.h"
#include "io/model_loader.h"
#include <cstring>
#include <unordered_map>
#include <functional>

namespace tts {
class Weights {
    Device device_;
    Quantization quantization_;
    SafeTensors files_;
    const std::vector<std::string> checkpoint_files_;
    std::unordered_map<std::string, Tensor> loaded_;
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
    TensorView allocate(const std::string &name, TensorShape shape, Type type) {
        require_registration();
        auto [parameter, inserted] = loaded_.try_emplace(name, shape, type, device_);
        if (!inserted) throw std::invalid_argument("duplicate parameter: " + name);
        return parameter->second.view();
    }
    static TensorShape checkpoint_shape(const SafeTensors::Entry &source) {
        const int rows = source.shape.empty() ? 1 : source.shape.front();
        if (rows < 1 || !source.size)
            throw std::invalid_argument("model parameter cannot be empty");
        const size_t row_bytes = size_t(rows) * bytes(source.type);
        const size_t columns = source.size / row_bytes;
        if (source.size % row_bytes || columns < 1 || columns > size_t(INT32_MAX))
            throw std::invalid_argument("weight exceeds tensor view dimensions");
        return {rows, int(columns)};
    }
    TensorView joined_shape(const std::vector<std::string> &parts) const {
        if (parts.empty()) throw std::invalid_argument("weight has no sources");
        const Type type = files_.get(parts.front()).type;
        int rows = 0, columns = 0;
        for (const auto &name : parts) {
            const auto &source = files_.get(name);
            const auto shape = checkpoint_shape(source);
            if (source.shape.empty() || source.type != type || shape.rows > INT32_MAX - rows ||
                (columns && columns != shape.cols))
                throw std::invalid_argument("incompatible weight sources: " + name);
            rows += shape.rows;
            columns = shape.cols;
        }
        return TensorView::metadata(rows, columns, type);
    }
    void copy_sources(TensorView destination, const std::vector<std::string> &parts) const {
        size_t offset = 0;
        for (const auto &name : parts) {
            const auto &source = files_.get(name);
            ops::upload_bytes(device_.index(),
                static_cast<char *>(destination.data) + offset, source.data, source.size);
            offset += source.size;
        }
    }

  public:
    // Layers first declare destination storage and load transforms. Execute
    // these once after the network is built, before any runner/graph exists.
    void load() {
        require_registration();
        phase_ = Phase::Loading;
        for (const auto &upload : uploads_) upload();
        for (const auto &process : postprocess_) process();
        ops::synchronize(device_.index());
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
    Device device() const { return device_; }
    Quantization quantization() const { return quantization_; }
    explicit Weights(Device device, Quantization quantization,
                      ModelLoader::Checkpoint checkpoint)
        : device_(device), quantization_(quantization), files_(std::move(checkpoint.tensors)),
          checkpoint_files_(std::move(checkpoint.files)) {}
    ~Weights() {
        // A failed load may still have H2D reads from the mapped checkpoint.
        // Pool frees are asynchronous; keep the mappings alive until reads finish.
        if (phase_ != Phase::Sealed) {
            try { ops::synchronize(device_.index()); } catch (...) {}
        }
    }
    TensorView get(const std::string &name) {
        auto it = loaded_.find(name);
        if (it != loaded_.end())
            return it->second.view();
        require_registration();
        const auto &source = files_.get(name);
        auto view = allocate(name, checkpoint_shape(source), source.type);
        uploads_.push_back([this, view, name] { copy_sources(view, {name}); });
        return view;
    }
    TensorView optional(const std::string &name) {
        auto it = loaded_.find(name);
        if (it != loaded_.end()) return it->second.view();
        require_loading();
        return files_.has(name) ? get(name) : TensorView{};
    }
    RowScaledView quantized(const std::string &name, const std::vector<std::string> &parts) {
        require_unique(name);
        require_unique(name + ".scales");
        const auto shape = joined_shape(parts);
        for (const auto &part : parts) {
            const auto &source = files_.get(part);
            const bool matrix = source.shape.size() == 2 ||
                (source.shape.size() == 3 && source.shape[2] == 1);
            if ((shape.type != Type::BF16 && shape.type != Type::F16) || !matrix)
                throw std::invalid_argument(
                    "quantized weights require compatible F16/BF16 matrices or pointwise projections");
        }
        RowScaledView result{allocate(name, shape, quantized_type(quantization_)),
                             allocate(name + ".scales", {shape.rows, 1}, Type::F32)};
        uploads_.push_back([this, parts, shape, result] {
            Tensor source(shape, shape.type, device_);
            copy_sources(source.view(), parts);
            ops::quantize_rows(source.view(), result);
            ops::synchronize(device_.index()); // Reuse a single matrix's 16-bit source capacity during loading.
        });
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
        const auto shape = joined_shape(parts);
        auto view = allocate(name, shape, shape.type);
        uploads_.push_back([this, parts, view] { copy_sources(view, parts); });
        return view;
    }
    TensorView custom(const std::string &name, const void *data, int rows, int cols, Type type) {
        require_unique(name);
        if (rows < 1 || cols < 1 || !data)
            throw std::invalid_argument("invalid custom weight");
        auto view = allocate(name, {rows, cols}, type);
        std::vector<char> packed(view.size());
        std::memcpy(packed.data(), data, packed.size());
        uploads_.push_back([this, view, packed = std::move(packed)] {
            ops::upload_bytes(device_.index(), view.data, packed.data(), packed.size());
        });
        return view;
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
        size_t total = 0;
        for (const auto &parameter : loaded_) total += parameter.second.view().size();
        return total;
    }
    const std::vector<std::string> &checkpoint_files() const { return checkpoint_files_; }
};
} // namespace tts
