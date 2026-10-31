# MSVC/CUDA compatibility for the pinned spdlog/fmt dependency.
get_target_property(spdlog_options spdlog_header_only INTERFACE_COMPILE_OPTIONS)
list(TRANSFORM spdlog_options REPLACE "^/Zc:__cplusplus$"
  "$<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>")
set_property(TARGET spdlog_header_only PROPERTY INTERFACE_COMPILE_OPTIONS "${spdlog_options}")

# Patch a build-local copy of fmt. C++ continues to use upstream headers;
# only NVCC sees this overlay. Never modify fetched/offline source trees.
find_package(Git REQUIRED)
set(fmt_source "${spdlog_SOURCE_DIR}/include/spdlog/fmt")
set(fmt_stage "${spdlog_BINARY_DIR}/fmt-patch-stage")
set(fmt_native "${spdlog_BINARY_DIR}/include/spdlog/fmt")
file(GLOB_RECURSE fmt_headers RELATIVE "${fmt_source}" "${fmt_source}/*")
foreach(header IN LISTS fmt_headers)
  configure_file("${fmt_source}/${header}" "${fmt_stage}/${header}" COPYONLY)
endforeach()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env
  --unset=GIT_DIR --unset=GIT_WORK_TREE --unset=GIT_INDEX_FILE --unset=GIT_COMMON_DIR
  "GIT_CEILING_DIRECTORIES=${fmt_stage}"
  "${GIT_EXECUTABLE}" apply --unsafe-paths
  "${CMAKE_CURRENT_LIST_DIR}/patches/fmt-nvcc-msvc.patch"
  WORKING_DIRECTORY "${fmt_stage}" COMMAND_ERROR_IS_FATAL ANY)
foreach(header IN LISTS fmt_headers)
  configure_file("${fmt_stage}/${header}" "${fmt_native}/${header}" COPYONLY)
endforeach()
target_include_directories(spdlog_header_only BEFORE INTERFACE
  "$<$<COMPILE_LANGUAGE:CUDA>:${spdlog_BINARY_DIR}/include>")
