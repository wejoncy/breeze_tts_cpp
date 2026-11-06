#pragma once
#include "ops/activation.h"
#include <cutlass/fast_math.h>
namespace tts::ops {
__device__ __forceinline__ float activate(float x, Activation kind, float alpha) {
    switch (kind) {
    case Activation::Relu:
        return fmaxf(x, 0.f);
    case Activation::LeakyRelu:
        return x >= 0 ? x : x * alpha;
    case Activation::Silu:
        return x / (1.f + expf(-x));
    case Activation::Gelu:
        return .5f * x * (1.f + erff(x * .7071067811865475f));
    case Activation::GeluTanh:
        return .5f * x * (1.f + cutlass::fast_tanh(.7978845608028654f * (x + .044715f * x * x * x)));
    case Activation::Mish:
        return x * tanhf(x > 20.f ? x : log1pf(expf(x)));
    case Activation::Tanh:
        return tanhf(x);
    case Activation::Sigmoid:
        return 1.f / (1.f + expf(-x));
    default:
        return x;
    }
}
}
