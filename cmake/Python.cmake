find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_VARIABLE tvm_ffi_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(tvm_ffi CONFIG REQUIRED)
if(WIN32)
  # The upstream package sets only IMPORTED_IMPLIB on Windows. Runtime metadata
  # also needs the actual DLL, discovered by the package's own locator.
  execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "from tvm_ffi.libinfo import find_libtvm_ffi; print(find_libtvm_ffi())"
    OUTPUT_VARIABLE tvm_ffi_RUNTIME OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  set_target_properties(tvm_ffi::shared PROPERTIES IMPORTED_LOCATION "${tvm_ffi_RUNTIME}")
endif()
add_library(tts_python SHARED bindings/tvm_ffi.cpp)
target_include_directories(tts_python PRIVATE src "${PROJECT_SOURCE_DIR}")
target_link_libraries(tts_python PRIVATE tts spdlog::spdlog_header_only tvm_ffi::shared)
set_target_properties(tts_python PROPERTIES INSTALL_RPATH "$ORIGIN")
tts_runtime_manifest(tts_python)
tts_deploy_runtime(tts_python)
install(TARGETS tts_python)
