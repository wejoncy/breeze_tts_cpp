#include "ops/basic.h"
#include <cstring>
#include "ops/storage.h"
#include "ops/sequence.h"
#include "ops/sampling.h"
#include "ops/kernels/quant/row_quantize.h"
#include "ops/kernels/quant/quant_norm.h"
#include "ops/kernels/quant/quant_gated.h"
#include "runtime/cuda_context.h"

namespace tts::ops {
void reserve_attention(int device, size_t elements) {
    auto &gpu = execution_context(device);
    if (elements > SIZE_MAX / sizeof(float)) throw std::overflow_error("attention workspace size");
    if (elements * sizeof(float) <= gpu.attention_lse.size()) return;
    if (gpu.plans_sealed) throw std::logic_error("attention workspace growth after initialization");
    gpu.memory.sync();
    Tensor::Scope allocation(gpu.memory);
    gpu.attention_lse = Buffer(elements * sizeof(float));
}
int decode_capacity(int device, int dimensions, Type type) {
    if ((type != Type::F16 && type != Type::BF16) || dimensions < 32 || dimensions > 128 || dimensions % 32)
        return 0;
    return execution_context(device).flash_decode_slots[(type == Type::F16 ? 4 : 0) + dimensions / 32 - 1];
}
void reserve_decode(int device, int dimensions, Type type) {
    auto &gpu = execution_context(device);
    const int slots = decode_capacity(device, dimensions, type);
    if (!slots) return;
    const size_t elements = size_t(slots) * dimensions;
    if (elements > INT_MAX) throw std::overflow_error("decode workspace capacity");
    const size_t output_bytes = elements * sizeof(float), lse_bytes = size_t(slots) * sizeof(float);
    if (output_bytes <= gpu.decode_output.size() && lse_bytes <= gpu.decode_lse.size()) return;
    if (gpu.plans_sealed) throw std::logic_error("decode workspace growth after initialization");
    gpu.memory.sync();
    Tensor::Scope allocation(gpu.memory);
    if (output_bytes > gpu.decode_output.size()) gpu.decode_output = Buffer(output_bytes);
    if (lse_bytes > gpu.decode_lse.size()) gpu.decode_lse = Buffer(lse_bytes);
}
int normalized_linear_capacity(TensorView input, bool qkv) {
    auto &gpu = execution_context(input);
    return qkv ? gpu.normalized_qkv_capacity : gpu.normalized_linear_capacity;
}
void zero(TensorView tensor) {
    if (!tensor.dense()) throw std::invalid_argument("zero requires dense storage");
    CUDA(cudaMemsetAsync(tensor.data, 0, tensor.size(), execution_context(tensor).stream));
}
void prepare_quantized(TensorView weight, int rows, int output_channels, int input_channels,
                        Type type, const Epilogue &epilogue, QuantizationGroups groups) {
    auto &quantized = *execution_context(weight).quantization;
    quantized.prepare(rows, output_channels, input_channels, type, groups);
    quantized.prepare_finish(rows, output_channels, input_channels, type, epilogue, groups);
}
void quantized_norm(TensorView input, TensorView update, TensorView weight, RowScaledView output,
                      float epsilon, bool gemma, TensorView scale) {
    const auto stream = execution_context(input, {update, weight, output.values, output.scales, scale}).stream;
    launch_quantized_norm(input, update, weight, output, epsilon, gemma, scale, stream);
}
void quantized_gated(TensorView input, RowScaledView output, int kind) {
    const auto &context = execution_context(input, {output.values, output.scales});
    launch_quantized_gated(input, output, kind, context.stream, context.multiprocessors,
                           context.quant_gated_clusters);
}
void matmul(RowScaledView input, RowScaledView weight, TensorView output) {
    execution_context(output, {input.values, input.scales, weight.values, weight.scales})
        .quantization->linear(input, weight, output);
}
void synchronize(int device) { if (device >= 0) execution_context(device).memory.sync(); }
void quantize_rows(TensorView input, RowScaledView output) {
    output.validate();
    if ((input.type != Type::BF16 && input.type != Type::F16) ||
        input.rows < 1 || input.cols < 1 || !input.data ||
        !output.values.data || !output.scales.data ||
        input.rows != output.values.rows || input.cols != output.values.cols)
        throw std::invalid_argument("quantization source/destination shape/type mismatch");
    const auto stream = execution_context(input, {output.values, output.scales}).stream;
    launch_row_quantize(input, output, stream);
}
} // namespace tts::ops
