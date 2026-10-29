# Executed after linking or during installation; no application code is loaded.
include("${TTS_RUNTIME_INFO}")
cmake_path(GET runtime_binary PARENT_PATH binary_dir)
cmake_path(GET runtime_binary FILENAME binary_name)
list(FILTER runtime_directories EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES runtime_directories)
# Windows searches beside the inspected binary first. Stage current target DLLs
# away from previously deployed copies so SDK updates cannot reuse stale codecs.
set(stage "${TTS_RUNTIME_INFO}.scan")
file(REMOVE_RECURSE "${stage}")
file(MAKE_DIRECTORY "${stage}")
file(COPY "${runtime_binary}" ${runtime_libraries} DESTINATION "${stage}")
list(REMOVE_ITEM runtime_directories "${binary_dir}")
file(GET_RUNTIME_DEPENDENCIES
  LIBRARIES "${stage}/${binary_name}"
  DIRECTORIES ${runtime_directories}
  PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-"
  RESOLVED_DEPENDENCIES_VAR dependencies)
file(MAKE_DIRECTORY "${TTS_RUNTIME_DESTINATION}")
file(TO_CMAKE_PATH "$ENV{SystemRoot}" system_root)
string(TOLOWER "${system_root}" system_root)
foreach(dependency IN LISTS dependencies)
  string(TOLOWER "${dependency}" normalized)
  if(system_root)
    cmake_path(IS_PREFIX system_root "${normalized}" NORMALIZE system_library)
    if(system_library)
      continue()
    endif()
  endif()
  cmake_path(GET dependency FILENAME name)
  file(COPY_FILE "${dependency}" "${TTS_RUNTIME_DESTINATION}/${name}" ONLY_IF_DIFFERENT)
endforeach()
file(REMOVE_RECURSE "${stage}")
