# Prefer package-provided targets. Runtime-only installations can use headers
# from the pinned upstream archive; no package manager or system writes occur.
function(tts_import_audio target prefix)
  if(WIN32)
    cmake_path(GET ${prefix}_LIBRARY PARENT_PATH library_dir)
    find_file(${prefix}_DLL NAMES ${ARGN}
      HINTS ${TTS_RUNTIME_DIRECTORIES} "${library_dir}" "${library_dir}/../bin"
      PATH_SUFFIXES bin REQUIRED)
    add_library(${target} SHARED IMPORTED GLOBAL)
    set_target_properties(${target} PROPERTIES
      IMPORTED_IMPLIB "${${prefix}_LIBRARY}"
      IMPORTED_LOCATION "${${prefix}_DLL}")
  else()
    add_library(${target} UNKNOWN IMPORTED GLOBAL)
    set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${${prefix}_LIBRARY}")
  endif()
  set_target_properties(${target} PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${${prefix}_INCLUDE_DIR}")
endfunction()

find_package(SndFile CONFIG QUIET)
if(NOT TARGET SndFile::sndfile)
  find_library(SNDFILE_LIBRARY NAMES sndfile sndfile-1 libsndfile-1 libsndfile.so.1 REQUIRED)
  find_path(SNDFILE_INCLUDE_DIR NAMES sndfile.h)
  if(NOT SNDFILE_INCLUDE_DIR)
    FetchContent_MakeAvailable(sndfile)
    set(SNDFILE_INCLUDE_DIR "${sndfile_SOURCE_DIR}/include")
  endif()
  tts_import_audio(SndFile::sndfile SNDFILE sndfile.dll sndfile-1.dll libsndfile.dll libsndfile-1.dll)
endif()

find_package(soxr CONFIG QUIET)
if(NOT TARGET soxr::soxr)
  find_library(SOXR_LIBRARY NAMES soxr soxr-0 libsoxr libsoxr.so.0 REQUIRED)
  find_path(SOXR_INCLUDE_DIR NAMES soxr.h)
  if(NOT SOXR_INCLUDE_DIR)
    FetchContent_MakeAvailable(soxr)
    set(SOXR_INCLUDE_DIR "${soxr_SOURCE_DIR}/src")
  endif()
  tts_import_audio(soxr::soxr SOXR soxr.dll soxr-0.dll libsoxr.dll libsoxr-0.dll)
endif()
