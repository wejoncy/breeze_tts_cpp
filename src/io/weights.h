#pragma once
#include "core/quantization.h"
#include "core/tensor_storage.h"
#include "io/model_loader.h"
#include <unordered_map>

namespace tts {
class Weights {
    Device device_;
    Quantization quantization_;
    SafeTensors files_;
    const std::vector<std::string> checkpoint_files_;
    std::unordered_map<std::string, Tensor> loaded_;
    struct Upload {
        enum class Kind { Copy, Quantize, Packed };
        Kind kind = Kind::Copy;
        TensorView destination;
        std::vector<std::string> sources;
        RowScaledView quantized;
        std::vector<char> packed;
    };
    std::vector<Upload> uploads_;
    enum class Phase { Registering, Loading, Loaded, Sealed };
    Phase phase_ = Phase::Registering;
    void require_loading() const;
    void require_unique(const std::string &name) const;
    void require_registration() const;
    TensorView allocate(const std::string &name, TensorShape shape, Type type);
    static TensorShape checkpoint_shape(const SafeTensors::Entry &source);
    TensorView joined_shape(const std::vector<std::string> &parts) const;
    void copy_sources(TensorView destination, const std::vector<std::string> &parts) const;

  public:
    // Layers first declare destination storage and load transforms. Execute
    // these once after the network is built, before any runner/graph exists.
    void load();
    void seal();
    Device device() const {
        return device_;
    }
    Quantization quantization() const {
        return quantization_;
    }
    explicit Weights(Device device, Quantization quantization, ModelLoader::Checkpoint checkpoint);
    ~Weights();
    TensorView get(const std::string &name);
    TensorView optional(const std::string &name);
    RowScaledView quantized(const std::string &name, const std::vector<std::string> &parts);
    const SafeTensors::Entry &entry(const std::string &name) const;
    void validate(const std::string &name, const std::vector<int> &shape, Type type) const;
    TensorView fused(const std::string &name, const std::vector<std::string> &parts);
    TensorView custom(const std::string &name, const void *data, int rows, int cols, Type type);
    TensorView generated(const std::string &name, const std::vector<float> &values, int rows,
                         int cols, Type type);
    TensorView matrix_columns(const std::string &name, const std::string &source, int first,
                              int count);
    std::vector<float> host_values(const std::string &name) const;
    size_t size() const;
    const std::vector<std::string> &checkpoint_files() const {
        return checkpoint_files_;
    }
};
} // namespace tts
