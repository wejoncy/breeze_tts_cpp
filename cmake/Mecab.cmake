# MeCab-Ko ships Meson/Autotools builds. Build its native library directly so
# the C++ engine does not acquire a Python/Meson build-time requirement.
function(tts_configure_mecab)
  include(CheckIncludeFileCXX)
  include(CheckCXXSymbolExists)
  include(CheckCXXSourceCompiles)
  include(CheckTypeSize)
  include(TestBigEndian)
  foreach(header IN ITEMS ctype.h dirent.h dlfcn.h fcntl.h inttypes.h io.h memory.h
      pthread.h stdint.h stdlib.h string.h strings.h sys/mman.h sys/stat.h
      sys/times.h sys/param.h sys/types.h unistd.h windows.h)
    string(MAKE_C_IDENTIFIER "HAVE_${header}" feature)
    string(TOUPPER "${feature}" feature)
    check_include_file_cxx("${header}" "${feature}")
  endforeach()
  check_cxx_symbol_exists(mmap sys/mman.h HAVE_MMAP)
  check_cxx_symbol_exists(getenv stdlib.h HAVE_GETENV)
  check_cxx_symbol_exists(getpagesize unistd.h HAVE_GETPAGESIZE)
  check_cxx_source_compiles("#include <sched.h>
int main() { int value=0; __sync_fetch_and_add(&value,1); sched_yield(); }" HAVE_GCC_ATOMIC_OPS)
  check_cxx_source_compiles("__thread int value=0; int main() { return value; }" HAVE_TLS_KEYWORD)
  set(CMAKE_EXTRA_INCLUDE_FILES stdlib.h sys/types.h)
  foreach(type IN ITEMS char int long "long long" short size_t)
    string(REPLACE " " "_" feature "${type}")
    string(TOUPPER "SIZEOF_${feature}" feature)
    check_type_size("${type}" "${feature}" LANGUAGE CXX)
  endforeach()
  check_type_size("unsigned long long int" HAVE_UNSIGNED_LONG_LONG_INT LANGUAGE CXX)
  test_big_endian(WORDS_BIGENDIAN)
  set(HAVE_LIBPTHREAD "${CMAKE_USE_PTHREADS_INIT}")
  set(HAVE_ICONV OFF) # All checkpoint dictionaries use UTF-8.
  set(PACKAGE)
  set(VERSION)
  file(READ "${mecab_SOURCE_DIR}/config.h.in" configuration)
  string(REGEX REPLACE "#undef ([A-Za-z0-9_]+)" "#cmakedefine \\1 @\\1@" configuration "${configuration}")
  file(MAKE_DIRECTORY "${mecab_BINARY_DIR}/source")
  file(WRITE "${mecab_BINARY_DIR}/config.h.in" "${configuration}")
  configure_file("${mecab_BINARY_DIR}/config.h.in" "${mecab_BINARY_DIR}/config.h")

  # A build-local header adds the conventional static-link guard on Windows;
  # neither a fetched source tree nor an offline source override is modified.
  file(GLOB files CONFIGURE_DEPENDS "${mecab_SOURCE_DIR}/src/*.cpp" "${mecab_SOURCE_DIR}/src/*.h")
  foreach(path IN LISTS files)
    cmake_path(GET path FILENAME name)
    if(NOT name STREQUAL "mecab.h" AND NOT name STREQUAL "dictionary.cpp")
      configure_file("${path}" "${mecab_BINARY_DIR}/source/${name}" COPYONLY)
    endif()
  endforeach()
  # MeCab uses binary_function without including its defining header.
  file(READ "${mecab_SOURCE_DIR}/src/dictionary.cpp" dictionary)
  file(CONFIGURE OUTPUT "${mecab_BINARY_DIR}/source/dictionary.cpp"
    CONTENT "#include <functional>\n${dictionary}" @ONLY)
  file(READ "${mecab_SOURCE_DIR}/src/mecab.h" header)
  string(REPLACE "#  ifdef DLL_EXPORT"
    "#  if defined(MECAB_STATIC)\n#    define MECAB_DLL_EXTERN extern\n#    define MECAB_DLL_CLASS_EXTERN\n#  elif defined(DLL_EXPORT)" header "${header}")
  file(CONFIGURE OUTPUT "${mecab_BINARY_DIR}/source/mecab.h" CONTENT "${header}" @ONLY)
  file(GLOB sources CONFIGURE_DEPENDS "${mecab_BINARY_DIR}/source/*.cpp")
  list(FILTER sources EXCLUDE REGEX "/mecab(-dict-index|-dict-gen|-cost-train|-system-eval|-test-gen)?\\.cpp$")
  add_library(tts_mecab STATIC ${sources})
  set_target_properties(tts_mecab PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 14)
  target_include_directories(tts_mecab PRIVATE "${mecab_BINARY_DIR}")
  target_include_directories(tts_mecab SYSTEM PUBLIC "${mecab_BINARY_DIR}/source")
  target_compile_definitions(tts_mecab PUBLIC MECAB_STATIC PRIVATE HAVE_CONFIG_H DIC_VERSION=102
    PACKAGE="mecab" VERSION="0.999" MECAB_DEFAULT_RC="" MECAB_USE_UTF8_ONLY)
  target_link_libraries(tts_mecab PRIVATE Threads::Threads)
endfunction()
tts_configure_mecab()
