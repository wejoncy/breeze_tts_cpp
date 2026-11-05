#pragma once
#include "io/json.h"
#include "tts/request.h"
#include <string>

namespace httplib {
struct Request;
}
namespace tts::http {
struct Payload {
    Json value;
    std::string uploaded_audio;
};
Payload parse_body(const httplib::Request &);
GenerationOptions parse_options(Json, GenerationOptions);
struct Speech {
    Input input;
    GenerationOptions options;
    std::string format;
    double speed = 1;
    bool stream = false, sse = false;
};
Speech parse_speech(Json &, GenerationOptions defaults);
std::string text_field(const Json &, const char *name, size_t limit,
                       const std::string &fallback = {});
std::string decode_reference(const Json &);
std::string pcm_bytes(const std::vector<int16_t> &);
std::string wav_bytes(const std::string &pcm, int sample_rate);
Json error_json(int status, const std::string &message, const Json &param = nullptr);
const char *finish_reason(const Stats &);
} // namespace tts::http
