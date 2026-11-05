#include "server/protocol.h"
#include "io/wav.h"
#include <httplib.h>
#include <cmath>
#include <cctype>
#include <limits>
#include <sstream>
#include <type_traits>

namespace tts::http {
namespace {
template <typename Value> Value number(const Json &value, const std::string &name) {
    bool valid = value.is_number() && !value.is_boolean();
    if constexpr (std::is_integral_v<Value>) {
        valid = value.is_number_integer() &&
                value.get<long double>() >= std::numeric_limits<Value>::lowest() &&
                value.get<long double>() <= std::numeric_limits<Value>::max();
    }
    if (!valid || !std::isfinite(value.get<double>()))
        throw std::invalid_argument("invalid " + name);
    auto result = value.get<Value>();
    if (!std::isfinite(static_cast<double>(result)))
        throw std::invalid_argument("out of range " + name);
    return result;
}
bool flag(const Json &value, const std::string &name) {
    if (value.is_boolean())
        return value.get<bool>();
    if (value.is_number_integer() && (value == 0 || value == 1))
        return value == 1;
    throw std::invalid_argument("invalid " + name);
}
} // namespace

std::string text_field(const Json &object, const char *name, size_t limit,
                       const std::string &fallback) {
    if (!object.contains(name))
        return fallback;
    const auto &value = object.at(name);
    if (!value.is_string())
        throw std::invalid_argument(std::string("invalid ") + name);
    auto text = value.get<std::string>();
    if (text.size() > limit || text.find('\0') != std::string::npos)
        throw std::invalid_argument(std::string("invalid ") + name);
    return text;
}

Payload parse_body(const httplib::Request &request) {
    if (request.body.empty() && !request.is_multipart_form_data())
        throw std::invalid_argument("request body is empty");
    Payload payload{Json::object(), {}};
    auto &value = payload.value;
    auto insert = [&](const std::string &name, const std::string &content) {
        if (name.empty() || value.contains(name))
            throw std::invalid_argument("invalid or duplicate form field");
        value[name] = content;
    };
    const auto type = request.get_header_value("Content-Type");
    if (request.is_multipart_form_data()) {
        auto upload = [&](const std::string &name, const std::string &content) {
            if (name != "ref_audio" || value.contains(name))
                throw std::invalid_argument("only one ref_audio upload is supported");
            insert(name, "");
            payload.uploaded_audio = content;
            if (payload.uploaded_audio.empty())
                throw std::invalid_argument("empty ref_audio upload");
        };
        for (const auto &[name, field] : request.form.fields) {
            if (name == "ref_audio")
                upload(name, field.content);
            else
                insert(name, field.content);
        }
        for (const auto &[name, file] : request.form.files)
            upload(name, file.content);
    } else if (type.find("application/x-www-form-urlencoded") == 0) {
        // httplib params also include the URL query; keep speech bodies unambiguous.
        if (request.target.find('?') != std::string::npos)
            throw std::invalid_argument("form parameters must be in the body");
        for (const auto &field : request.params)
            insert(field.first, field.second);
    } else {
        value = parse_json(request.body);
        if (!value.is_object())
            throw std::invalid_argument("JSON object required");
        return payload;
    }
    for (const auto *name : {"items",
                             "references",
                             "speed",
                             "max_frames",
                             "max_new_tokens",
                             "seed",
                             "top_k",
                             "sample_steps",
                             "depth_top_k",
                             "depth_decoder_top_k",
                             "temperature",
                             "speed_factor",
                             "streaming_chunk_seconds",
                             "streaming_lookahead_seconds",
                             "streaming_overlap_seconds",
                             "semantic_lookahead_seconds",
                             "gpt_streaming_lookahead_seconds",
                             "fragment_interval",
                             "cfg_rate",
                             "top_p",
                             "cfg_scale",
                             "cfg_scale_ref",
                             "cfg_scale_ins",
                             "repetition_penalty",
                             "depth_temperature",
                             "depth_top_p",
                             "depth_decoder_temperature",
                             "depth_decoder_top_p"})
        if (value.contains(name))
            value[name] = parse_json(value.at(name).get<std::string>());
    for (const auto *name : {"stream", "do_sample", "depth_do_sample", "depth_decoder_do_sample",
                             "use_cuda_graph", "dual_cfg"}) {
        if (!value.contains(name))
            continue;
        auto text = value.at(name).get<std::string>();
        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char ch) { return char(std::tolower(ch)); });
        if (text != "true" && text != "false" && text != "1" && text != "0")
            throw std::invalid_argument(std::string("invalid ") + name);
        value[name] = text == "true" || text == "1";
    }
    return payload;
}

GenerationOptions parse_options(Json values, GenerationOptions options) {
    for (const auto &alias : {std::pair{"max_new_tokens", "max_frames"},
                              {"streaming_window_mode", "synthesis_mode"},
                              {"gpt_streaming_lookahead_seconds", "semantic_lookahead_seconds"},
                              {"depth_decoder_temperature", "depth_temperature"},
                              {"depth_decoder_top_k", "depth_top_k"},
                              {"depth_decoder_top_p", "depth_top_p"},
                              {"depth_decoder_do_sample", "depth_do_sample"}}) {
        if (!values.contains(alias.first))
            continue;
        if (values.contains(alias.second))
            throw std::invalid_argument(std::string("conflicting option aliases: ") + alias.second);
        values[alias.second] = values.at(alias.first);
    }
    for (const auto *name : {"temperature", "top_k", "top_p", "do_sample"}) {
        auto depth = std::string("depth_") + name;
        if (values.contains(name) && !values.contains(depth))
            values[depth] = values.at(name);
    }
    auto set = [&](const char *name, auto &destination) {
        if (!values.contains(name) || values.at(name).is_null())
            return;
        using Value = std::decay_t<decltype(destination)>;
        if constexpr (std::is_same_v<Value, bool>)
            destination = flag(values.at(name), name);
        else
            destination = number<Value>(values.at(name), name);
    };
    set("max_frames", options.max_frames);
    set("top_k", options.top_k);
    set("temperature", options.temperature);
    set("cfg_scale", options.cfg_scale);
    set("seed", options.seed);
    set("use_cuda_graph", options.use_cuda_graph);
    set("top_p", options.top_p);
    set("repetition_penalty", options.repetition_penalty);
    set("do_sample", options.do_sample);
    set("depth_top_k", options.depth_top_k);
    set("depth_temperature", options.depth_temperature);
    set("depth_top_p", options.depth_top_p);
    set("depth_do_sample", options.depth_do_sample);
    set("dual_cfg", options.dual_cfg);
    set("cfg_scale_ref", options.cfg_scale_ref);
    set("cfg_scale_ins", options.cfg_scale_ins);
    set("speed_factor", options.speed_factor);
    set("sample_steps", options.sample_steps);
    set("cfg_rate", options.cfg_rate);
    set("streaming_chunk_seconds", options.streaming_chunk_seconds);
    set("streaming_lookahead_seconds", options.streaming_lookahead_seconds);
    set("streaming_overlap_seconds", options.streaming_overlap_seconds);
    set("semantic_lookahead_seconds", options.semantic_lookahead_seconds);
    if (values.contains("synthesis_mode"))
        options.synthesis_mode = parse_synthesis_mode(text_field(values, "synthesis_mode", 16));
    if (values.contains("cfg_scale_ref") != values.contains("cfg_scale_ins"))
        throw std::invalid_argument("both cfg_scale_ref and cfg_scale_ins are required");
    if (values.contains("cfg_scale_ref"))
        options.dual_cfg = true;
    auto sampling_valid = [](float temperature, int top_k, float top_p) {
        return temperature >= 0 && top_k >= 0 && top_p > 0 && top_p <= 1;
    };
    if (options.max_frames < 1 || options.cfg_scale < 0 || options.repetition_penalty <= 0 ||
        options.speed_factor < .1 || options.speed_factor > 4 || options.sample_steps < 0 ||
        options.cfg_rate < -1 ||
        !sampling_valid(options.temperature, options.top_k, options.top_p) ||
        !sampling_valid(options.depth_temperature, options.depth_top_k, options.depth_top_p))
        throw std::invalid_argument("generation option outside supported range");
    if (values.contains("stream") && !values.at("stream").is_boolean())
        throw std::invalid_argument("stream must be boolean");
    return options;
}

Input parse_input(const Json &value) {
    if (!value.is_object())
        throw std::invalid_argument("each request must be an object");
    Input input;
    input.text = text_field(value, "input", 65536);
    if (input.text.find_first_not_of(" \r\n\t") == std::string::npos)
        throw std::invalid_argument("input is required");
    input.instruction = text_field(value, "instructions", 16384);
    auto voice = text_field(value, "voice", 256, "default");
    if (voice != "default")
        input.speaker = std::move(voice);
    input.language = text_field(value, "language", 32);
    input.text_split_method = text_field(value, "text_split_method", 8);
    if (!input.text_split_method.empty() &&
        (input.text_split_method.size() != 4 || input.text_split_method.substr(0, 3) != "cut" ||
         input.text_split_method[3] < '0' || input.text_split_method[3] > '5'))
        throw std::invalid_argument("invalid text_split_method");
    auto pause = value.value("fragment_interval", Json(0));
    input.fragment_interval = number<double>(pause, "fragment_interval");
    if (input.fragment_interval < 0 || input.fragment_interval > 10)
        throw std::invalid_argument("invalid fragment_interval");
    return input;
}

Speech parse_speech(Json &value, GenerationOptions defaults) {
    if (!value.is_object())
        throw std::invalid_argument("speech item must be an object");
    // These controls require model capabilities this engine does not expose.
    for (const char *name : {"task_type", "x_vector_only_mode", "initial_codec_chunk_frames",
                             "stage_params", "token_count", "duration_tokens",
                             "stream_codec_output", "suppress_bootstrap_silence", "epsilon_cutoff"})
        if (value.contains(name) && !value.at(name).is_null())
            throw std::invalid_argument(std::string("unsupported control: ") + name);
    for (const char *name : {"instructions", "language", "ref_text", "model"})
        if (value.contains(name) && value.at(name).is_null())
            value.erase(name);
    for (const char *name : {"language", "prompt_language"}) {
        if (!value.contains(name))
            continue;
        auto language = text_field(value, name, 32);
        std::transform(language.begin(), language.end(), language.begin(),
                       [](unsigned char ch) { return char(std::tolower(ch)); });
        for (const auto &label : {std::pair{"chinese", "zh"},
                                  {"english", "en"},
                                  {"japanese", "ja"},
                                  {"korean", "ko"},
                                  {"cantonese", "yue"},
                                  {"german", "de"},
                                  {"french", "fr"},
                                  {"russian", "ru"},
                                  {"portuguese", "pt"},
                                  {"spanish", "es"},
                                  {"italian", "it"}})
            if (language == label.first) {
                language = label.second;
                break;
            }
        value[name] = std::move(language);
    }
    Speech speech;
    speech.input = parse_input(value);
    speech.options = parse_options(value, defaults);
    if (value.contains("speed"))
        speech.speed = number<double>(value.at("speed"), "speed");
    if (speech.speed < .25 || speech.speed > 4)
        throw std::invalid_argument("speed must be between 0.25 and 4");
    speech.format = text_field(value, "response_format", 16, "wav");
    if (speech.format != "wav" && speech.format != "pcm")
        throw std::invalid_argument("response_format must be wav or pcm");
    auto transport = text_field(value, "stream_format", 16, "audio");
    if (transport != "audio" && transport != "sse")
        throw std::invalid_argument("stream_format must be audio or sse");
    speech.sse = transport == "sse";
    speech.stream = value.value("stream", false) || speech.sse;
    if (speech.stream && speech.format != "pcm")
        throw std::invalid_argument("streaming requires response_format='pcm'");
    if (value.contains("references") && !value.at("references").is_null()) {
        const auto &references = value.at("references");
        if (!references.is_array() || references.size() != 1 || !references[0].is_object())
            throw std::invalid_argument("references must contain exactly one audio reference");
        if ((value.contains("ref_audio") && !value.at("ref_audio").is_null()) ||
            (value.contains("ref_text") && !value.at("ref_text").is_null()))
            throw std::invalid_argument("use either references or ref_audio/ref_text");
        const auto &reference = references[0];
        if (!reference.contains("audio_path") || reference.contains("speaker_embedding") ||
            reference.contains("vq_codes"))
            throw std::invalid_argument("reference requires audio_path and text");
        value["ref_audio"] = reference.at("audio_path");
        value["ref_text"] = text_field(reference, "text", 65536);
    }
    return speech;
}

std::string decode_reference(const Json &value) {
    if (value.is_null())
        return {};
    if (!value.is_string())
        throw std::invalid_argument("ref_audio must be base64 encoded audio");
    auto encoded = value.get<std::string>();
    if (encoded.rfind("data:", 0) == 0) {
        auto comma = encoded.find(',');
        if (comma == std::string::npos ||
            encoded.substr(0, comma).find(";base64") == std::string::npos)
            throw std::invalid_argument("reference data URL must contain base64 audio");
        encoded.erase(0, comma + 1);
    }
    if (encoded.find("://") != std::string::npos)
        throw std::invalid_argument(
            "reference URLs are unsupported; use base64, a data URL or a multipart upload");
    if (encoded.size() % 4)
        throw std::invalid_argument("invalid base64 ref_audio");
    std::string decoded;
    decoded.reserve(encoded.size() / 4 * 3);
    auto digit = [](char ch) -> int {
        if (ch >= 'A' && ch <= 'Z')
            return ch - 'A';
        if (ch >= 'a' && ch <= 'z')
            return ch - 'a' + 26;
        if (ch >= '0' && ch <= '9')
            return ch - '0' + 52;
        if (ch == '+')
            return 62;
        if (ch == '/')
            return 63;
        return -1;
    };
    for (size_t offset = 0; offset < encoded.size(); offset += 4) {
        int first = digit(encoded[offset]), second = digit(encoded[offset + 1]);
        bool pad2 = encoded[offset + 2] == '=', pad3 = encoded[offset + 3] == '=';
        int third = pad2 ? 0 : digit(encoded[offset + 2]);
        int fourth = pad3 ? 0 : digit(encoded[offset + 3]);
        if (first < 0 || second < 0 || third < 0 || fourth < 0 || (pad2 && !pad3) ||
            ((pad2 || pad3) && offset + 4 != encoded.size()) || (pad2 && (second & 15)) ||
            (pad3 && !pad2 && (third & 3)))
            throw std::invalid_argument("invalid base64 ref_audio");
        decoded.push_back(char((first << 2) | (second >> 4)));
        if (!pad2)
            decoded.push_back(char((second << 4) | (third >> 2)));
        if (!pad3)
            decoded.push_back(char((third << 6) | fourth));
    }
    return decoded;
}

std::string pcm_bytes(const std::vector<int16_t> &samples) {
    if (samples.empty())
        return {};
    return {reinterpret_cast<const char *>(samples.data()), samples.size() * sizeof(int16_t)};
}
std::string wav_bytes(const std::string &pcm, int sample_rate) {
    std::ostringstream stream(std::ios::binary | std::ios::out);
    write_wav(stream, pcm, sample_rate);
    return stream.str();
}
Json error_json(int status, const std::string &message, const Json &param) {
    const char *type = status == 400   ? "BadRequestError"
                       : status == 404 ? "NotFoundError"
                       : status == 503 ? "ServiceUnavailableError"
                                       : "InternalServerError";
    return {{"error", {{"message", message}, {"type", type}, {"param", param}, {"code", status}}}};
}
const char *finish_reason(const Stats &stats) {
    return stats.truncated ? "length" : "stop";
}
} // namespace tts::http
