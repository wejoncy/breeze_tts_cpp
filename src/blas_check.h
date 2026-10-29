#pragma once
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
namespace tts {
inline void check(cublasStatus_t s, const char *expr) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(expr) + ": cuBLAS status " + std::to_string(s));
}
} // namespace tts
#define BLAS(x) ::tts::check((x), #x)
