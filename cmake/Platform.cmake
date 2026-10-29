# Host compiler policy belongs here; CUDA architecture selection stays on targets.
if(MSVC)
  add_compile_definitions(_USE_MATH_DEFINES)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8;/Zc:__cplusplus;/bigobj>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/Zc:__cplusplus,/bigobj>")
endif()
set(TTS_RUNTIME_DIRECTORIES "" CACHE STRING "Additional dependency DLL directories")
if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN SPDLOG_WCHAR_FILENAMES)
endif()

function(tts_test_assertions target)
  if(MSVC)
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/UNDEBUG>")
  else()
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-UNDEBUG>")
  endif()
  target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>")
endfunction()

function(tts_runtime_manifest target)
  set(runtime_directories "[]")
  foreach(directory IN LISTS CUDAToolkit_BIN_DIR TTS_RUNTIME_DIRECTORIES)
    if(directory)
      string(REPLACE "\\" "/" directory "${directory}")
      string(REPLACE "\"" "\\\"" directory "${directory}")
      string(JSON index LENGTH "${runtime_directories}")
      string(JSON runtime_directories SET "${runtime_directories}" ${index} "\"${directory}\"")
    endif()
  endforeach()
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/runtime.json.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${target}-runtime.json.in" @ONLY)
  file(GENERATE OUTPUT "$<TARGET_FILE_DIR:${target}>/tts-runtime.json"
    INPUT "${CMAKE_CURRENT_BINARY_DIR}/${target}-runtime.json.in")
  include(GNUInstallDirs)
  if(WIN32)
    set(destination "${CMAKE_INSTALL_BINDIR}")
  else()
    set(destination "${CMAKE_INSTALL_LIBDIR}")
  endif()
  install(FILES "$<TARGET_FILE_DIR:${target}>/tts-runtime.json" DESTINATION "${destination}")
endfunction()

# Keep native and Python artifacts runnable from the build/install directory.
# Inspect binaries as well as CMake targets: codec DLLs may be private dependencies
# of a prebuilt audio library and therefore absent from TARGET_RUNTIME_DLLS.
function(tts_deploy_runtime target)
  if(NOT WIN32)
    return()
  endif()
  if(MSVC)
    cmake_path(GET CMAKE_LINKER PARENT_PATH linker_dir)
    find_program(runtime_inspector NAMES dumpbin HINTS "${linker_dir}" NO_CACHE REQUIRED)
    set(runtime_tool dumpbin)
  else()
    set(runtime_inspector "${CMAKE_OBJDUMP}")
    set(runtime_tool objdump)
  endif()
  set(info "${CMAKE_CURRENT_BINARY_DIR}/runtime/$<CONFIG>/${target}.cmake")
  file(GENERATE OUTPUT "${info}" CONTENT
"set(runtime_binary [==[$<TARGET_FILE:${target}>]==])
set(runtime_libraries [==[$<TARGET_RUNTIME_DLLS:${target}>]==])
set(runtime_directories [==[$<TARGET_RUNTIME_DLL_DIRS:${target}>;${TTS_RUNTIME_DIRECTORIES};${CUDAToolkit_BIN_DIR}]==])
set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM windows+pe)
set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL ${runtime_tool})
set(CMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND [==[${runtime_inspector}]==])
")
  set(deploy "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DeployRuntime.cmake")
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" "-DTTS_RUNTIME_INFO=${info}"
      "-DTTS_RUNTIME_DESTINATION=$<TARGET_FILE_DIR:${target}>" -P "${deploy}"
    VERBATIM)
  include(GNUInstallDirs)
  install(CODE "set(TTS_RUNTIME_INFO \"${info}\")
set(TTS_RUNTIME_DESTINATION \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\")
include(\"${deploy}\")")
endfunction()
