include_guard(GLOBAL)
include(FetchContent)

# Archive digests lock the tested sources. Standard FetchContent source overrides
# and FETCHCONTENT_FULLY_DISCONNECTED support pre-populated offline builds.
FetchContent_Declare(json
  URL https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.12.0
  URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187
  EXCLUDE_FROM_ALL)
FetchContent_Declare(spdlog
  URL https://codeload.github.com/gabime/spdlog/tar.gz/refs/tags/v1.15.3
  URL_HASH SHA256=15a04e69c222eb6c01094b5c7ff8a249b36bb22788d72519646fb85feb267e67
  EXCLUDE_FROM_ALL)
FetchContent_Declare(concurrentqueue
  URL https://codeload.github.com/cameron314/concurrentqueue/tar.gz/refs/tags/v1.0.4
  URL_HASH SHA256=87fbc9884d60d0d4bf3462c18f4c0ee0a9311d0519341cac7cbd361c885e5281
  EXCLUDE_FROM_ALL
  SOURCE_SUBDIR _tts_headers_only)
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
FetchContent_Declare(corrosion
  URL https://codeload.github.com/corrosion-rs/corrosion/tar.gz/refs/tags/v0.5.2
  URL_HASH SHA256=6bc02411e29183a896aa60c58db6819ec6cf57c08997481d0b0da9029356b529
  EXCLUDE_FROM_ALL)
FetchContent_Declare(cutlass
  URL https://codeload.github.com/NVIDIA/cutlass/tar.gz/dc4817921edda44a549197ff3a9dcf5df0636e7b
  URL_HASH SHA256=f2a3a9df5e6f010c8b02716aa2644a6f071827fafa606fac5f5241cab6a1ab56
  EXCLUDE_FROM_ALL
  SOURCE_SUBDIR _tts_headers_only)
FetchContent_Declare(flash_attention
  URL https://codeload.github.com/Dao-AILab/flash-attention/tar.gz/060c9188beec3a8b62b33a3bfa6d5d2d44975fab
  URL_HASH SHA256=64391cfec73fc5a25932b2c0c2fb3b4d9f97af001164e5220af21049fc17e04e
  EXCLUDE_FROM_ALL
  SOURCE_SUBDIR _tts_headers_only)
FetchContent_Declare(sndfile
  URL https://codeload.github.com/libsndfile/libsndfile/tar.gz/refs/tags/1.2.2
  URL_HASH SHA256=ffe12ef8add3eaca876f04087734e6e8e029350082f3251f565fa9da55b52121
  EXCLUDE_FROM_ALL
  SOURCE_SUBDIR _tts_headers_only)
FetchContent_Declare(soxr
  URL https://codeload.github.com/chirlu/soxr/tar.gz/refs/tags/0.1.3
  URL_HASH SHA256=db6ca1b1e8405c6ef92f8294fc123d910abf0a114003b3f0f13fa57a95fd62d0
  EXCLUDE_FROM_ALL
  SOURCE_SUBDIR _tts_headers_only)

function(tts_fetch_dependencies)
  # Keep upstream examples, tests, and install rules out of this project.
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(JSON_BuildTests OFF)
  set(JSON_Install OFF)
  set(JSON_MultipleHeaders OFF)
  set(SPDLOG_BUILD_EXAMPLE OFF)
  set(SPDLOG_BUILD_TESTS OFF)
  set(SPDLOG_INSTALL OFF)
  set(CORROSION_BUILD_TESTS OFF)
  set(CORROSION_DEV_MODE OFF)
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_PROGRAMS OFF)
  FetchContent_MakeAvailable(json spdlog concurrentqueue cppjieba fasttext corrosion cutlass flash_attention)
  FetchContent_MakeAvailable(openjtalk mecab)
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

  if(MSVC)
    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Spdlog.cmake")
  endif()

  add_library(tts_concurrentqueue INTERFACE)
  target_include_directories(tts_concurrentqueue SYSTEM INTERFACE "${concurrentqueue_SOURCE_DIR}")
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

  add_library(tts_cutlass INTERFACE)
  target_include_directories(tts_cutlass SYSTEM INTERFACE "${cutlass_SOURCE_DIR}/include")
  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/FlashAttention.cmake")
  include(GNUInstallDirs)
  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Audio.cmake")

  install(FILES "${json_SOURCE_DIR}/LICENSE.MIT"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/json")
  install(FILES "${cutlass_SOURCE_DIR}/LICENSE.txt"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/cutlass")
  install(FILES "${concurrentqueue_SOURCE_DIR}/LICENSE.md"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/concurrentqueue")
  install(FILES "${cppjieba_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/cppjieba")
  install(FILES "${fasttext_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/fasttext")
  install(FILES "${openjtalk_SOURCE_DIR}/src/COPYING"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/openjtalk")
  install(FILES "${mecab_SOURCE_DIR}/BSD" "${mecab_SOURCE_DIR}/COPYING"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/mecab")
  foreach(name IN ITEMS spdlog corrosion flash_attention)
    install(FILES "${${name}_SOURCE_DIR}/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/${name}")
  endforeach()
  install(FILES "${PROJECT_SOURCE_DIR}/licenses/linguistics.txt"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/linguistics")
endfunction()
tts_fetch_dependencies()
