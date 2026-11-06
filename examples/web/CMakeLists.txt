set(web_page "${CMAKE_CURRENT_SOURCE_DIR}/index.html")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${web_page}")
file(READ "${web_page}" TTS_WEB_HTML)
configure_file(web_page.h.in generated/web_page.h @ONLY)

target_sources(tts_server PRIVATE web.cpp voice_presets.cpp)
target_include_directories(tts_server PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/generated")
