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
add_library(tts_text OBJECT
  src/text/frontend.cpp
  src/text/language_router.cpp
  src/text/g2pw.cpp
  src/text/english.cpp
  src/text/cantonese.cpp
  src/text/morphology.cpp
  src/text/korean.cpp
  src/text/japanese.cpp
  src/text/language_detector.cpp
  src/text/budoux.cpp
  src/text/segmentation.cpp
  src/text/jieba.cpp
  src/text/pinyin.cpp
  src/text/tone_sandhi.cpp
  src/text/normalization.cpp
  src/text/phonemizer.cpp
  src/text/regex.cpp
)
set_target_properties(tts_text PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tts_text PRIVATE include src "${PROJECT_SOURCE_DIR}")
target_link_libraries(tts_text PRIVATE nlohmann_json::nlohmann_json tts_cppjieba tts_fasttext openjtalk tts_mecab)
add_library(tts SHARED
  bindings/c_api.cpp
  src/ops/kernels/tensor_kernels.cu
  src/ops/cuda/lt_linear.cpp
  src/ops/cuda/grouped_linear.cpp
  src/ops/cuda/relative_attention.cpp
  src/ops/cuda/fft.cpp
  src/ops/cuda/execution_context.cpp
  src/ops/linear.cpp
  src/ops/basic.cpp
  src/ops/tensor.cpp
  src/runtime/cuda_context.cpp
  src/ops/convolution.cpp
  src/ops/kernels/sequence/sequence_pointwise.cu
  src/ops/kernels/sequence/sequence_norm.cu
  src/ops/kernels/sequence/sequence_conditioning.cu
  src/layers/convolution.cpp
  src/layers/transformer.cpp
  src/models/bert/model.cpp
  src/layers/spectrogram.cpp
  src/execution/model_registry.cpp
  src/models/gpt_sovits/model.cpp
  src/models/g2pw/model.cpp
  src/models/gpt_sovits/vocoder.cpp
  src/models/gpt_sovits/dit.cpp
  src/models/gpt_sovits/cfm.cpp
  src/models/hubert/model.cpp
  src/models/gpt_sovits/semantic.cpp
  src/layers/transformer/relative.cpp
  src/layers/audio/style_encoder.cpp
  src/layers/audio/wavenet.cpp
  src/models/gpt_sovits/acoustic.cpp
  src/models/gpt_sovits/reference_encoder.cpp
  src/execution/model_runtime.cpp
  src/execution/input_staging.cpp
  src/execution/input_packing.cpp
  src/execution/model_runner.cpp
  src/inputs/input_registry.cpp
  src/inputs/token_inputs.cpp
  src/inputs/phoneme_inputs.cpp
  src/engine/engine.cpp
  src/engine/request_executor.cpp
  src/engine/request_scheduler.cpp
  src/io/audio_io.cpp
  src/io/mapped_file.cpp
  src/io/model_loader.cpp
  src/io/weights.cpp
  src/io/safetensors.cpp
  src/ops/kernels/attention/flash_attention.cu
  src/ops/kernels/attention/flash_decoding.cu
  src/ops/kernels/attention/relative_attention.cu
  src/ops/kernels/audio/audio_kernels.cu
  src/ops/kernels/audio/encoder_kernels.cu
  src/ops/kernels/audio/spectral_kernels.cu
  src/ops/kernels/quant/quantized_gemm.cu
  src/ops/kernels/quant/quant_dispatch.cu
  src/ops/kernels/quant/quant_gated.cu
  src/ops/kernels/quant/quant_gemm_sm80.cu
  src/ops/kernels/quant/quant_norm.cu
  src/ops/kernels/quant/vector_quantize.cu
  src/ops/kernels/sampling/sampling.cu
  src/ops/kernels/text/pronunciation.cu
  src/models/mimi/encoder.cpp
  src/models/mimi/decoder.cpp
  src/models/breeze/model.cpp
  src/models/breeze/model_config.cpp
  src/support/log.cpp
)
option(TTS_BUILD_HOPPER_KERNELS "Build native SM90a quantized GEMM kernels" ON)
target_compile_definitions(tts PRIVATE TTS_HAVE_HOPPER_KERNELS=$<BOOL:${TTS_BUILD_HOPPER_KERNELS}>)
if(TTS_BUILD_HOPPER_KERNELS)
  add_library(tts_quant_hopper OBJECT src/ops/kernels/quant/quant_gemm_sm90.cu)
  set_target_properties(tts_quant_hopper PROPERTIES CUDA_ARCHITECTURES 90a POSITION_INDEPENDENT_CODE ON)
  target_include_directories(tts_quant_hopper PRIVATE include src "${PROJECT_SOURCE_DIR}")
  target_link_libraries(tts_quant_hopper PRIVATE tts_cutlass)
  target_compile_options(tts_quant_hopper PRIVATE --expt-relaxed-constexpr -lineinfo)
  target_sources(tts PRIVATE $<TARGET_OBJECTS:tts_quant_hopper>)
endif()
target_include_directories(tts PUBLIC include PRIVATE src "${PROJECT_SOURCE_DIR}")
target_link_libraries(tts PRIVATE tts_text nlohmann_json::nlohmann_json spdlog::spdlog_header_only tts_concurrentqueue tts_flash_attention tts_hf_tokenizer SndFile::sndfile soxr::soxr CUDA::cublas CUDA::cublasLt CUDA::cufft CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
target_compile_options(tts PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_source_files_properties(
  src/ops/kernels/tensor_kernels.cu
  src/ops/kernels/quant/quantized_gemm.cu
  src/ops/kernels/quant/quant_norm.cu
  src/ops/kernels/quant/quant_gated.cu
  src/ops/kernels/quant/quant_dispatch.cu
  src/ops/kernels/audio/encoder_kernels.cu
  src/ops/kernels/audio/audio_kernels.cu
  src/ops/kernels/audio/spectral_kernels.cu
  src/ops/kernels/sampling/sampling.cu
  src/ops/kernels/quant/vector_quantize.cu
  PROPERTIES COMPILE_OPTIONS "--fmad=false")
set_source_files_properties(
  src/ops/kernels/attention/flash_attention.cu
  src/ops/kernels/attention/relative_attention.cu
  src/ops/kernels/quant/quant_gemm_sm80.cu
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
