cmake_minimum_required(VERSION 3.28)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures (native detects the local GPU)")
endif()
project(tts_cpp VERSION 0.1.0 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
include(cmake/Platform.cmake)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
include(cmake/Dependencies.cmake)
corrosion_import_crate(MANIFEST_PATH tokenizer/Cargo.toml LOCKED)
add_library(tts SHARED
  bindings/c_api.cpp
  kernels/backend.cu
  src/engine/engine.cpp
  src/engine/request_executor.cpp
  src/engine/request_scheduler.cpp
  src/execution/model_runner.cpp
  src/execution/prefill.cpp
  src/execution/prepared_batch.cpp
  src/execution/request_planner.cpp
  src/io/audio_io.cpp
  src/io/mapped_file.cpp
  src/io/safetensors.cpp
  kernels/attention/flash_attention.cu
  kernels/audio/audio_kernels.cu
  kernels/audio/encoder_kernels.cu
  kernels/quant/quant_backend.cu
  kernels/quant/quant_dispatch.cu
  kernels/quant/quant_gated.cu
  kernels/quant/quant_gemm_sm80.cu
  kernels/quant/quant_norm.cu
  kernels/sampling/sampling.cu
  src/models/breeze/audio_encoder.cpp
  src/models/breeze/codec.cpp
  src/models/breeze/model.cpp
  src/models/breeze/model_config.cpp
  src/support/log.cpp
)
option(TTS_BUILD_HOPPER_KERNELS "Build native SM90a quantized GEMM kernels" ON)
target_compile_definitions(tts PRIVATE TTS_HAVE_HOPPER_KERNELS=$<BOOL:${TTS_BUILD_HOPPER_KERNELS}>)
if(TTS_BUILD_HOPPER_KERNELS)
  add_library(tts_quant_hopper OBJECT kernels/quant/quant_gemm_sm90.cu)
  set_target_properties(tts_quant_hopper PROPERTIES CUDA_ARCHITECTURES 90a POSITION_INDEPENDENT_CODE ON)
  target_include_directories(tts_quant_hopper PRIVATE include src "${PROJECT_SOURCE_DIR}")
  target_link_libraries(tts_quant_hopper PRIVATE tts_cutlass)
  target_compile_options(tts_quant_hopper PRIVATE --expt-relaxed-constexpr -lineinfo)
  target_sources(tts PRIVATE $<TARGET_OBJECTS:tts_quant_hopper>)
endif()
target_include_directories(tts PUBLIC include PRIVATE src "${PROJECT_SOURCE_DIR}")
target_link_libraries(tts PRIVATE nlohmann_json::nlohmann_json spdlog::spdlog_header_only tts_concurrentqueue tts_flash_attention tts_hf_tokenizer SndFile::sndfile soxr::soxr CUDA::cublas CUDA::cublasLt CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(tts PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_source_files_properties(
  kernels/backend.cu
  kernels/quant/quant_backend.cu
  kernels/quant/quant_norm.cu
  kernels/quant/quant_gated.cu
  kernels/quant/quant_dispatch.cu
  kernels/audio/encoder_kernels.cu
  kernels/audio/audio_kernels.cu
  kernels/sampling/sampling.cu
  PROPERTIES COMPILE_OPTIONS "--fmad=false")
set_source_files_properties(
  kernels/attention/flash_attention.cu
  kernels/quant/quant_gemm_sm80.cu
  PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr")
set_target_properties(tts PROPERTIES POSITION_INDEPENDENT_CODE ON WINDOWS_EXPORT_ALL_SYMBOLS ON)
tts_deploy_runtime(tts)
add_executable(tts_cli src/cli/main.cpp src/cli/cli_platform.cpp)
set_target_properties(tts_cli PROPERTIES OUTPUT_NAME tts ARCHIVE_OUTPUT_NAME tts_cli)
target_include_directories(tts_cli PRIVATE src "${PROJECT_SOURCE_DIR}")
target_link_libraries(tts_cli PRIVATE nlohmann_json::nlohmann_json spdlog::spdlog_header_only tts tts_hf_tokenizer CUDA::cublas CUDA::cublasLt CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
option(TTS_BUILD_PYTHON "Build the TVM FFI Python bridge" OFF)
if(TTS_BUILD_PYTHON)
  include(cmake/Python.cmake)
endif()
install(TARGETS tts tts_cli)
install(DIRECTORY include/tts DESTINATION include FILES_MATCHING PATTERN "*.h")
