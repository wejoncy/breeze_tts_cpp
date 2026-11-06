#include "web.h"
#include "voice_presets.h"
#include "web_page.h"
#include "server/protocol.h"
#include "support/log.h"
#include <httplib.h>
#include <memory>

namespace tts::http {
namespace {
void reply(httplib::Response &response, int status, const Json &body) {
    response.status = status;
    response.set_content(body.dump(), "application/json; charset=utf-8");
}
} // namespace

void mount_web_ui(httplib::Server &http, const std::string &voice_presets_file) {
    http.Get("/", [](const httplib::Request &, httplib::Response &response) {
        response.set_content(web_page, sizeof(web_page) - 1, "text/html; charset=utf-8");
    });
    auto storage = std::make_shared<VoicePresets>(voice_presets_file);
    auto presets = [storage](const httplib::Request &request, httplib::Response &response) {
        try {
            if (request.method == "GET" || request.method == "HEAD") {
                reply(response, 200, {{"enabled", storage->enabled()}, {"presets", storage->list()}});
            } else {
                if (!storage->enabled()) {
                    reply(response, 503, error_json(503, "voice preset storage is not configured"));
                    return;
                }
                if (request.method == "POST") {
                    reply(response, 200, storage->save(parse_json(request.body)));
                } else {
                    if (storage->erase(request.matches[1].str()))
                        reply(response, 200, {{"deleted", true}});
                    else
                        reply(response, 404, error_json(404, "voice preset not found"));
                }
            }
        } catch (const std::out_of_range &error) {
            reply(response, 404, error_json(404, error.what()));
        } catch (const std::invalid_argument &error) {
            reply(response, 400, error_json(400, error.what()));
        } catch (const Json::exception &error) {
            reply(response, 400, error_json(400, error.what()));
        } catch (const std::exception &error) {
            TTS_LOG(TTS_LOG_ERROR, "http", "voice presets: {}", error.what());
            reply(response, 500, error_json(500, error.what()));
        }
    };
    http.Get("/v1/voice-presets", presets);
    http.Post("/v1/voice-presets", presets);
    http.Delete(R"(/v1/voice-presets/([a-f0-9]{32}))", presets);
}
} // namespace tts::http
