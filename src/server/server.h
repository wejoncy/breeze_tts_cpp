#pragma once
#include <memory>
#include <string>

namespace tts {
class Engine;
namespace http {
struct Config {
    std::string host = "127.0.0.1";
    std::string model_name;
    std::string voice_presets_file;
    int port = 8088;
    size_t threads = 64;
    size_t queue_limit = 32;
    size_t stream_buffer_bytes = 1 << 20;
    int request_timeout_seconds = 300;
    size_t batch_max_items = 32;
    void validate() const;
};
class Server {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    Server(Engine &, const Config &);
    ~Server();
    // bind before listen lets startup failures reach the owner synchronously.
    void bind();
    // The owner starts Engine::run_async() before listen accepts requests.
    void listen();
    // Stop ingress and cancel HTTP-owned requests; listen joins all handlers.
    void stop();
};
} // namespace http
} // namespace tts
