# Benchmarks reuse the engine targets and the same dependency graph.
foreach(name IN ITEMS kernels kv_pages fp8_operators quant_kernels selection_regret fp8_rms fp8_gated)
  if(EXISTS "${PROJECT_SOURCE_DIR}/bench/${name}.cpp")
    set(source "bench/${name}.cpp")
  else()
    set(source "bench/${name}.cu")
  endif()
  add_executable(${name} EXCLUDE_FROM_ALL ${source})
  target_include_directories(${name} PRIVATE src)
  target_link_libraries(${name} PRIVATE tts tts_cutlass nlohmann_json::nlohmann_json
    CUDA::cublas CUDA::cublasLt CUDA::cudart)
  target_compile_options(${name} PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--fmad=false;-lineinfo>")
endforeach()

set(TTS_HOPPOU_SOURCE_DIR "" CACHE PATH "Optional Breeze-TTS-2.cpp comparison source")
set(TTS_HOPPOU_BUILD_DIR "" CACHE PATH "Optional Breeze-TTS-2.cpp comparison build")
if(TTS_HOPPOU_SOURCE_DIR)
  if(NOT TTS_HOPPOU_BUILD_DIR)
    set(TTS_HOPPOU_BUILD_DIR "${TTS_HOPPOU_SOURCE_DIR}/build-cuda")
  endif()
  add_executable(compare_hoppou EXCLUDE_FROM_ALL bench/compare_hoppou.cpp)
  target_include_directories(compare_hoppou PRIVATE src "${TTS_HOPPOU_SOURCE_DIR}/include"
    "${TTS_HOPPOU_SOURCE_DIR}/third_party/ggml/include")
  target_link_libraries(compare_hoppou PRIVATE tts nlohmann_json::nlohmann_json
    CUDA::cublas CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
  foreach(name IN ITEMS breeze_core ggml ggml-cpu ggml-base ggml-cuda)
    find_library(TTS_HOPPOU_${name} NAMES ${name}
      PATHS "${TTS_HOPPOU_BUILD_DIR}" "${TTS_HOPPOU_BUILD_DIR}/third_party/ggml/src"
        "${TTS_HOPPOU_BUILD_DIR}/third_party/ggml/src/ggml-cuda"
      NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(compare_hoppou PRIVATE "${TTS_HOPPOU_${name}}")
  endforeach()
endif()
