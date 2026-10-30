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
  FetchContent_MakeAvailable(json spdlog concurrentqueue corrosion cutlass flash_attention)

  add_library(tts_concurrentqueue INTERFACE)
  target_include_directories(tts_concurrentqueue SYSTEM INTERFACE "${concurrentqueue_SOURCE_DIR}")

  add_library(tts_cutlass INTERFACE)
  target_include_directories(tts_cutlass SYSTEM INTERFACE "${cutlass_SOURCE_DIR}/include")
  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/FlashAttention.cmake")
  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Audio.cmake")

  include(GNUInstallDirs)
  install(FILES "${json_SOURCE_DIR}/LICENSE.MIT"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/json")
  install(FILES "${cutlass_SOURCE_DIR}/LICENSE.txt"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/cutlass")
  install(FILES "${concurrentqueue_SOURCE_DIR}/LICENSE.md"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/concurrentqueue")
  foreach(name IN ITEMS spdlog corrosion flash_attention)
    install(FILES "${${name}_SOURCE_DIR}/LICENSE"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/${name}")
  endforeach()
endfunction()
tts_fetch_dependencies()
