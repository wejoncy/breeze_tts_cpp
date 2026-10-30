cmake_minimum_required(VERSION 3.28)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 90 CACHE STRING "CUDA architectures")
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
  src/backends/cuda/backend.cu
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
  src/kernels/attention/flash_attention.cu
  src/kernels/audio/audio_kernels.cu
  src/kernels/audio/encoder_kernels.cu
  src/kernels/quant/quant_backend.cu
  src/kernels/quant/quant_dispatch.cu
  src/kernels/quant/quant_gated.cu
  src/kernels/quant/quant_gemm_sm80.cu
  src/kernels/quant/quant_norm.cu
  src/kernels/sampling/sampling.cu
  src/models/breeze/audio_encoder.cpp
  src/models/breeze/codec.cpp
  src/models/breeze/model.cpp
  src/models/breeze/model_config.cpp
  src/support/log.cpp
)
add_library(tts_quant_hopper OBJECT src/kernels/quant/quant_gemm_sm90.cu)
set_target_properties(tts_quant_hopper PROPERTIES CUDA_ARCHITECTURES 90a POSITION_INDEPENDENT_CODE ON)
target_include_directories(tts_quant_hopper PRIVATE include src)
target_link_libraries(tts_quant_hopper PRIVATE tts_cutlass)
target_compile_options(tts_quant_hopper PRIVATE --expt-relaxed-constexpr -lineinfo)
target_sources(tts PRIVATE $<TARGET_OBJECTS:tts_quant_hopper>)
target_include_directories(tts PUBLIC include PRIVATE src)
target_link_libraries(tts PRIVATE nlohmann_json::nlohmann_json spdlog::spdlog_header_only tts_concurrentqueue tts_flash_attention tts_hf_tokenizer SndFile::sndfile soxr::soxr CUDA::cublas CUDA::cublasLt CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(tts PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_source_files_properties(
  src/backends/cuda/backend.cu
  src/kernels/quant/quant_backend.cu
  src/kernels/quant/quant_norm.cu
  src/kernels/quant/quant_gated.cu
  src/kernels/quant/quant_dispatch.cu
  src/kernels/audio/encoder_kernels.cu
  src/kernels/audio/audio_kernels.cu
  src/kernels/sampling/sampling.cu
  PROPERTIES COMPILE_OPTIONS "--fmad=false")
set_source_files_properties(
  src/kernels/attention/flash_attention.cu
  src/kernels/quant/quant_gemm_sm80.cu
  PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr")
set_target_properties(tts PROPERTIES POSITION_INDEPENDENT_CODE ON WINDOWS_EXPORT_ALL_SYMBOLS ON)
tts_deploy_runtime(tts)
add_executable(tts_cli src/cli/main.cpp src/cli/cli_platform.cpp)
set_target_properties(tts_cli PROPERTIES OUTPUT_NAME tts)
target_include_directories(tts_cli PRIVATE src)
target_link_libraries(tts_cli PRIVATE nlohmann_json::nlohmann_json spdlog::spdlog_header_only tts tts_hf_tokenizer CUDA::cublas CUDA::cublasLt CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
option(TTS_BUILD_PYTHON "Build the TVM FFI Python bridge" OFF)
if(TTS_BUILD_PYTHON)
  include(cmake/Python.cmake)
endif()
install(TARGETS tts tts_cli)
install(DIRECTORY include/tts DESTINATION include FILES_MATCHING PATTERN "*.h")
include(CTest)
if(BUILD_TESTING)
  include(cmake/Tests.cmake)
endif()
include(cmake/Benchmarks.cmake)
