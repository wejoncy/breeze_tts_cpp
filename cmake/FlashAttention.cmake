# Patch a build-local copy, never the fetched source or an offline source override.
# copy_if_different keeps configure-only runs from recompiling attention.
find_package(Git REQUIRED)
set(flash_patch "${CMAKE_CURRENT_LIST_DIR}/patches/flash-attention-native.patch")
set(flash_stage "${flash_attention_BINARY_DIR}/patch-stage")
set(flash_native "${flash_attention_BINARY_DIR}/include/flash-attention")
file(MAKE_DIRECTORY "${flash_stage}" "${flash_native}")
file(GLOB flash_headers "${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/*.h"
  "${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/*.cuh")
foreach(header IN LISTS flash_headers)
  cmake_path(GET header FILENAME name)
  configure_file("${header}" "${flash_stage}/${name}" COPYONLY)
endforeach()
execute_process(COMMAND "${GIT_EXECUTABLE}" apply --unsafe-paths
  "--directory=${flash_stage}" "${flash_patch}" COMMAND_ERROR_IS_FATAL ANY)
foreach(header IN LISTS flash_headers)
  cmake_path(GET header FILENAME name)
  configure_file("${flash_stage}/${name}" "${flash_native}/${name}" COPYONLY)
endforeach()
add_library(tts_flash_attention INTERFACE)
target_include_directories(tts_flash_attention SYSTEM INTERFACE
  "${flash_attention_BINARY_DIR}/include")
target_link_libraries(tts_flash_attention INTERFACE tts_cutlass)
