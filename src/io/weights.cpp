#include "io/weights.h"

namespace tts {
TensorView Weights::generated(const std::string &name, const std::vector<float> &values,
                              int rows, int cols, Type type) {
    require_loading();
    if (values.size() != size_t(rows) * cols)
        throw std::invalid_argument("generated weight shape");
    if (type == Type::F32) return custom(name, values.data(), rows, cols, type);
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
