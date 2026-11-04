#pragma once
#include "models/bert/model.h"

namespace tts {
class Pronunciation {
    Bert bert_;
    Linear classifier_, pos_classifier_;
    TensorView descriptor_bias_, char_descriptor_, second_order_descriptor_, legal_;
  public:
    Pronunciation(Weights &, const std::string &prefix, const Json &metadata);
    TensorView forward(TensorInputs, ForwardContext &) const;
};
} // namespace tts
