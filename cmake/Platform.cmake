# Host compiler policy belongs here; CUDA architecture selection stays on targets.
if(MSVC)
  if(CMAKE_GENERATOR MATCHES "^Ninja")
    # Ninja compares /showIncludes prefixes as bytes. CMake's locale decoding
    # can corrupt a UTF-8 prefix on a Chinese Windows installation.
    set(msvc_probe_dir "${CMAKE_BINARY_DIR}/CMakeFiles/tts-msvc-deps")
    file(MAKE_DIRECTORY "${msvc_probe_dir}")
    file(WRITE "${msvc_probe_dir}/prefix.h" "// Dependency prefix probe.\n")
    file(WRITE "${msvc_probe_dir}/prefix.cpp" "#include \"prefix.h\"\n")
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" /nologo /utf-8 /showIncludes /EP
      "${msvc_probe_dir}/prefix.cpp"
      OUTPUT_VARIABLE msvc_probe ERROR_VARIABLE msvc_probe
      ENCODING NONE COMMAND_ERROR_IS_FATAL ANY)
    string(REPLACE "\\" "/" msvc_probe "${msvc_probe}")
    string(REGEX MATCH "[^\r\n]*prefix\\.h" msvc_include_line "${msvc_probe}")
    string(TOLOWER "${msvc_include_line}" msvc_include_lower)
    string(TOLOWER "${msvc_probe_dir}/prefix.h" msvc_probe_path)
    string(FIND "${msvc_include_lower}" "${msvc_probe_path}" msvc_path_start)
    if(msvc_path_start LESS 1)
      message(FATAL_ERROR "Cannot detect MSVC /showIncludes prefix")
    endif()
    string(SUBSTRING "${msvc_include_line}" 0 ${msvc_path_start} CMAKE_CL_SHOWINCLUDES_PREFIX)
  endif()
  add_compile_definitions(_USE_MATH_DEFINES)
  # CUDA 13/CCCL requires the conforming MSVC preprocessor.
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8;/Zc:__cplusplus;/Zc:preprocessor;/bigobj>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/Zc:__cplusplus,/Zc:preprocessor,/bigobj>")
endif()
set(TTS_RUNTIME_DIRECTORIES "" CACHE STRING "Additional dependency DLL directories")
if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN SPDLOG_WCHAR_FILENAMES)
endif()

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
