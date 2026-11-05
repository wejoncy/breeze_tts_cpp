include(FetchContent)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
FetchContent_Declare(httplib
  URL https://codeload.github.com/yhirose/cpp-httplib/tar.gz/refs/tags/v0.64.0
  URL_HASH SHA256=1ada413782af5e67e80452c39b7c90beb6fa4fc8341e18c0896bf95684b9a620
  EXCLUDE_FROM_ALL)
# HTTP needs no TLS/compression discovery or Python code generation.
set(HTTPLIB_COMPILE OFF)
set(HTTPLIB_INSTALL OFF)
set(HTTPLIB_TEST OFF)
foreach(feature OPENSSL WOLFSSL MBEDTLS ZLIB BROTLI ZSTD)
  set(HTTPLIB_USE_${feature}_IF_AVAILABLE OFF)
endforeach()
FetchContent_MakeAvailable(httplib)

add_executable(tts_server src/server/main.cpp src/server/server.cpp
  src/server/protocol.cpp src/support/main.cpp)
set_target_properties(tts_server PROPERTIES OUTPUT_NAME tts-server)
target_compile_definitions(tts_server PRIVATE TTS_ENTRY_POINT=run_server)
target_include_directories(tts_server PRIVATE src)
target_link_libraries(tts_server PRIVATE tts httplib::httplib
  nlohmann_json::nlohmann_json spdlog::spdlog_header_only Threads::Threads)
tts_deploy_runtime(tts_server)
install(TARGETS tts_server)
install(FILES "${httplib_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/tts.cpp/httplib")
