#pragma once
#include <stdexcept>

namespace tts {
struct DeviceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct DeviceOutOfMemory : DeviceError {
    using DeviceError::DeviceError;
};
} // namespace tts
