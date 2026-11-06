#include "io/weights.h"
#include "ops/storage.h"
#include "support/init_log.h"
#include "support/log.h"
#include <cstring>

namespace tts {
void Weights::require_loading() const {
    if (phase_ == Phase::Sealed)
        throw std::logic_error("weight allocation/upload after model initialization");
}
void Weights::require_unique(const std::string &name) const {
    require_registration();
    if (loaded_.count(name) || files_.has(name))
        throw std::invalid_argument("duplicate derived parameter: " + name);
}
void Weights::require_registration() const {
    if (phase_ != Phase::Registering)
        throw std::logic_error("parameter registration after weight loading started");
}
TensorView Weights::allocate(const std::string &name, TensorShape shape, Type type) {
    require_registration();
    auto [parameter, inserted] = loaded_.try_emplace(name, shape, type, device_);
    if (!inserted)
        throw std::invalid_argument("duplicate parameter: " + name);
    return parameter->second.view();
}
TensorShape Weights::checkpoint_shape(const SafeTensors::Entry &source) {
    const int rows = source.shape.empty() ? 1 : source.shape.front();
    if (rows < 1 || !source.size)
        throw std::invalid_argument("model parameter cannot be empty");
    const size_t row_bytes = size_t(rows) * bytes(source.type);
    const size_t columns = source.size / row_bytes;
    if (source.size % row_bytes || columns < 1 || columns > size_t(INT32_MAX))
        throw std::invalid_argument("weight exceeds tensor view dimensions");
    return {rows, int(columns)};
}
TensorView Weights::joined_shape(const std::vector<std::string> &parts) const {
    if (parts.empty())
        throw std::invalid_argument("weight has no sources");
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
void Weights::copy_sources(TensorView destination, const std::vector<std::string> &parts) const {
    int row = 0;
    for (const auto &name : parts) {
        const auto &source = files_.get(name);
        const int count = checkpoint_shape(source).rows;
        ops::upload_bytes(destination.slice(row, count), source.data, source.size);
        row += count;
    }
}
void Weights::load() {
    require_registration();
    phase_ = Phase::Loading;
    log::InitPhase phase("weight upload and quantization submission");
    TTS_LOG(TTS_LOG_INFO, "engine.init",
            "loading weights={:.0f} MiB quantization={}", size() / (1024.0 * 1024.0),
            quantization_name(quantization_));
    for (const auto &upload : uploads_)
        upload();
    phase.next("weight postprocessing and upload completion");
    for (const auto &process : postprocess_)
        process();
    ops::synchronize(device_.index());
    uploads_.clear();
    postprocess_.clear();
    phase_ = Phase::Loaded;
}
void Weights::after_loading(std::function<void()> process) {
    require_registration();
    postprocess_.push_back(std::move(process));
}
void Weights::seal() {
    require_loading();
    if (phase_ != Phase::Loaded)
        throw std::logic_error("model weights have not been loaded");
    // load() has synchronized every upload and load-time transform.
    files_ = SafeTensors{};
    phase_ = Phase::Sealed;
}
Weights::Weights(Device device, Quantization quantization, ModelLoader::Checkpoint checkpoint)
    : device_(device), quantization_(quantization), files_(std::move(checkpoint.tensors)),
      checkpoint_files_(std::move(checkpoint.files)) {}
Weights::~Weights() {
    // A failed load may still have H2D reads from the mapped checkpoint.
    // Pool frees are asynchronous; keep the mappings alive until reads finish.
    if (phase_ != Phase::Sealed) {
        try {
            ops::synchronize(device_.index());
        } catch (...) {
        }
    }
}
TensorView Weights::get(const std::string &name) {
    auto it = loaded_.find(name);
    if (it != loaded_.end())
        return it->second.view();
    require_registration();
    const auto &source = files_.get(name);
    auto view = allocate(name, checkpoint_shape(source), source.type);
    uploads_.push_back([this, view, name] { copy_sources(view, {name}); });
    return view;
}
TensorView Weights::optional(const std::string &name) {
    auto it = loaded_.find(name);
    if (it != loaded_.end())
        return it->second.view();
    require_loading();
    return files_.has(name) ? get(name) : TensorView{};
}
RowScaledView Weights::quantized(const std::string &name, const std::vector<std::string> &parts) {
    const auto layout = quantization_groups(quantization_);
    require_unique(name);
    require_unique(name + ".scales");
    const auto shape = joined_shape(parts);
    for (const auto &part : parts) {
        const auto &source = files_.get(part);
        const bool matrix =
            source.shape.size() == 2 || (source.shape.size() == 3 && source.shape[2] == 1);
        if ((shape.type != Type::BF16 && shape.type != Type::F16) || !matrix)
            throw std::invalid_argument(
                "quantized weights require compatible F16/BF16 matrices or pointwise projections");
    }
    RowScaledView result{allocate(name, shape, quantized_type(quantization_)),
                         allocate(name + ".scales",
                                  quantization_scale_shape(shape.rows, shape.cols, layout.weight),
                                  Type::F32),
                         layout.weight, layout.rotation};
    result.validate();
    uploads_.push_back([this, parts, shape, result] {
        Tensor source(shape, shape.type, device_);
        copy_sources(source.view(), parts);
        ops::quantize_rows(source.view(), result);
        ops::synchronize(
            device_.index()); // Reuse a single matrix's 16-bit source capacity during loading.
    });
    return result;
}
const SafeTensors::Entry &Weights::entry(const std::string &name) const {
    require_loading();
    return files_.get(name);
}
void Weights::validate(const std::string &name, const std::vector<int> &shape, Type type) const {
    const auto &value = entry(name);
    if (value.shape != shape || value.type != type)
        throw std::invalid_argument("checkpoint parameter shape/type mismatch: " + name);
}
TensorView Weights::fused(const std::string &name, const std::vector<std::string> &parts) {
    require_unique(name);
    const auto shape = joined_shape(parts);
    auto view = allocate(name, shape, shape.type);
    uploads_.push_back([this, parts, view] { copy_sources(view, parts); });
    return view;
}
TensorView Weights::custom(const std::string &name, const void *data, int rows, int cols,
                           Type type) {
    require_unique(name);
    if (rows < 1 || cols < 1 || !data)
        throw std::invalid_argument("invalid custom weight");
    auto view = allocate(name, {rows, cols}, type);
    std::vector<char> packed(view.size());
    std::memcpy(packed.data(), data, packed.size());
    uploads_.push_back([this, view, packed = std::move(packed)] {
        ops::upload_bytes(view, packed.data(), packed.size());
    });
    return view;
}
TensorView Weights::matrix_columns(const std::string &name, const std::string &source, int first,
                                   int count) {
    require_loading();
    const auto &e = entry(source);
    if (e.shape.size() != 2 || first < 0 || count < 1 || first > e.shape[1] - count)
        throw std::invalid_argument("weight column partition");
    std::vector<char> packed(size_t(e.shape[0]) * count * bytes(e.type));
    for (int row = 0; row < e.shape[0]; ++row)
        std::memcpy(packed.data() + size_t(row) * count * bytes(e.type),
                    static_cast<const char *>(e.data) +
                        (size_t(row) * e.shape[1] + first) * bytes(e.type),
                    size_t(count) * bytes(e.type));
    return custom(name, packed.data(), e.shape[0], count, e.type);
}
size_t Weights::size() const {
    size_t total = 0;
    for (const auto &parameter : loaded_)
        total += parameter.second.view().size();
    return total;
}
TensorView Weights::generated(const std::string &name, const std::vector<float> &values, int rows,
                              int cols, Type type) {
    require_loading();
    if (values.size() != size_t(rows) * cols)
        throw std::invalid_argument("generated weight shape");
    if (type == Type::F32)
        return custom(name, values.data(), rows, cols, type);
    Tensor packed({rows, cols}, type);
    ops::convert_host({const_cast<float *>(values.data()), rows, cols, Type::F32}, packed.view());
    return custom(name, packed.data_ptr(), rows, cols, type);
}
std::vector<float> Weights::host_values(const std::string &name) const {
    const auto &source = entry(name);
    const auto shape = checkpoint_shape(source);
    std::vector<float> values(shape.numel());
    ops::convert_host({const_cast<void *>(source.data), shape.rows, shape.cols, source.type},
                      {values.data(), shape.rows, shape.cols, Type::F32});
    return values;
}
} // namespace tts
