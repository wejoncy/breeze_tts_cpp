FetchContent_Declare(cppjieba
  URL https://codeload.github.com/yanyiwu/cppjieba/tar.gz/b9c787eb7750a8449400e7d7f59cf86237c23154
  URL_HASH SHA256=185b2018f4bf34d90027465426fee3424d0f8ed1aeebb967f20738557d0ed05c
  EXCLUDE_FROM_ALL
  SOURCE_SUBDIR _tts_headers_only)
FetchContent_Declare(fasttext
  URL https://codeload.github.com/facebookresearch/fastText/tar.gz/1142dc4c4ecbc19cc16eee5cdd28472e689267e6
  URL_HASH SHA256=c8b6cf146839a2716658fa5517a0810beed482caf2fc0bd2f63df10088d8d507
  EXCLUDE_FROM_ALL
  SOURCE_SUBDIR _tts_library_only)
FetchContent_Declare(openjtalk
  URL https://codeload.github.com/r9y9/open_jtalk/tar.gz/462fc38e7520aa89e4d32b2611749208528c901e
  URL_HASH SHA256=c2d9c5e83d8ae7c9f8564f73224523a70bab3e10aa808f60609a6ed59d3f9db3
  EXCLUDE_FROM_ALL
  SOURCE_SUBDIR _tts_library_only)
FetchContent_Declare(mecab
  URL https://codeload.github.com/Pusnow/mecab-ko-msvc/tar.gz/398f5f550045bebcfdef3c756cca302c659967df
  URL_HASH SHA256=5ebde3563b94b7dcd8ffc28a230b0c4c5f8cad1fe3950ea60ce6bcf93855094c
  EXCLUDE_FROM_ALL
  SOURCE_SUBDIR _tts_library_only)

FetchContent_MakeAvailable(cppjieba fasttext openjtalk mecab)
# Only the linguistic C library is needed. Its upstream build configures a
# second MeCab and writes generated headers into the fetched source tree.
file(GLOB openjtalk_sources CONFIGURE_DEPENDS "${openjtalk_SOURCE_DIR}/src/*/*.c")
list(FILTER openjtalk_sources EXCLUDE REGEX "/bin/")
add_library(openjtalk STATIC ${openjtalk_sources})
set_target_properties(openjtalk PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(openjtalk PRIVATE CHARSET_UTF_8)
foreach(source IN LISTS openjtalk_sources)
  cmake_path(GET source PARENT_PATH directory)
  target_include_directories(openjtalk SYSTEM PUBLIC "${directory}")
endforeach()
# MeCab-Ko also reads Japanese dictionaries, retaining Korean space penalties.
include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Mecab.cmake")

add_library(tts_cppjieba INTERFACE)
target_include_directories(tts_cppjieba SYSTEM INTERFACE "${cppjieba_SOURCE_DIR}/include")
# Upstream's top-level build adds CLI/training targets and global GNU flags.
# Compile its library sources with the project's portable compiler policy.
file(GLOB fasttext_sources CONFIGURE_DEPENDS "${fasttext_SOURCE_DIR}/src/*.cc")
list(REMOVE_ITEM fasttext_sources "${fasttext_SOURCE_DIR}/src/main.cc")
add_library(tts_fasttext STATIC ${fasttext_sources})
set_target_properties(tts_fasttext PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tts_fasttext SYSTEM PUBLIC "${fasttext_SOURCE_DIR}/src")
target_link_libraries(tts_fasttext PRIVATE Threads::Threads)

install(FILES "${cppjieba_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/cppjieba")
install(FILES "${fasttext_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/fasttext")
install(FILES "${openjtalk_SOURCE_DIR}/src/COPYING"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/openjtalk")
install(FILES "${mecab_SOURCE_DIR}/BSD" "${mecab_SOURCE_DIR}/COPYING"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/mecab")
install(FILES "${PROJECT_SOURCE_DIR}/licenses/linguistics.txt"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/linguistics")
