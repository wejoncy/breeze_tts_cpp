#include "io/weights.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tts {
TensorView Weights::generated(const std::string &name, const std::vector<float> &values,
                     int rows, int cols, Type type) {
    require_loading();
    if (values.size() != size_t(rows) * cols)
        throw std::invalid_argument("generated weight shape");
    if (type == Type::F32) return custom(name, values.data(), rows, cols, type);
    std::vector<uint16_t> packed(values.size());
    for (size_t element_index = 0; element_index < values.size(); ++element_index) {
        if (type == Type::F16) {
            __half value = __float2half_rn(values[element_index]);
            std::memcpy(&packed[element_index], &value, sizeof(value));
        } else if (type == Type::BF16) {
            __nv_bfloat16 value = __float2bfloat16_rn(values[element_index]);
            std::memcpy(&packed[element_index], &value, sizeof(value));
        } else throw std::invalid_argument("generated weight requires floating point");
    }
    return custom(name, packed.data(), rows, cols, type);
}
std::vector<float> Weights::host_values(const std::string &name) const {
    const auto &source_entry = entry(name);
    std::vector<float> values(source_entry.size / bytes(source_entry.type));
    for (size_t element_index = 0; element_index < values.size(); ++element_index) {
        if (source_entry.type == Type::F32) values[element_index] = static_cast<const float *>(source_entry.data)[element_index];
        else if (source_entry.type == Type::F16) values[element_index] = float(static_cast<const __half *>(source_entry.data)[element_index]);
        else if (source_entry.type == Type::BF16) values[element_index] = float(static_cast<const __nv_bfloat16 *>(source_entry.data)[element_index]);
        else throw std::invalid_argument("floating point checkpoint values required");
    }
    return values;
}
} // namespace tts
