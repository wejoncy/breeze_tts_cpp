#pragma once
#include <string>

namespace httplib { class Server; }
namespace tts::http {
void mount_web_ui(httplib::Server &, const std::string &voice_presets_file);
}
