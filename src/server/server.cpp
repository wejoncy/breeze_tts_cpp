#include "server/server.h"
#include "server/protocol.h"
#include "tts/engine.h"
#include "io/audio_io.h"
#include "io/audio_speed.h"
#include "support/log.h"
#include <httplib.h>
#include <atomic>
#include <chrono>
#include <list>
#include <mutex>
#include <deque>

namespace tts::http {
namespace {
using Clock = std::chrono::steady_clock;
struct HttpError : std::runtime_error {
    int status;
    HttpError(int code, const std::string &message) : std::runtime_error(message), status(code) {}
};
void json_reply(httplib::Response &response, int status, const Json &value) {
    response.status = status;
    response.set_content(value.dump(), "application/json");
}
Json api_error(std::exception_ptr error) {
    try {
        std::rethrow_exception(error);
    } catch (const HttpError &failure) {
        return error_json(failure.status, failure.what());
    } catch (const RequestOverloaded &failure) {
        return error_json(503, failure.what());
    } catch (const Json::exception &failure) {
        return error_json(400, failure.what());
    } catch (const std::invalid_argument &failure) {
        return error_json(400, failure.what());
    } catch (const std::exception &failure) {
        TTS_LOG(TTS_LOG_ERROR, "http", "inference: {}", failure.what());
        return error_json(422, failure.what());
    } catch (...) {
        return error_json(500, "internal server error");
    }
}
} // namespace

void Config::validate() const {
    if (threads < 2 || !queue_limit || !stream_buffer_bytes || port < 1 || port > 65535 ||
        request_timeout_seconds < 1 || !batch_max_items || host.empty())
        throw std::invalid_argument("invalid HTTP server limits");
}

struct Server::Impl {
    Engine &engine;
    Config config;
    AudioFormat audio;
    httplib::Server http;
    std::atomic<bool> stopping{false};
    // HTTP ownership only. Never hold this mutex across an Engine call or socket IO.
    std::mutex ownership;
    size_t active = 0;
    uint64_t sequence = 0;
    struct CachedReference {
        std::string audio, text, language;
        std::shared_ptr<const Reference> reference;
    };
    std::list<CachedReference> references;
    size_t cached_bytes = 0;
    std::deque<std::pair<std::string, Json>> outcomes;

    struct Job {
        Impl &owner;
        Request request;
        Clock::time_point deadline;
        bool submitted = false, terminal = false;
        Speech speech;
        Stats stats;
        std::unique_ptr<AudioSpeed> speed;
        explicit Job(Impl &server) : owner(server) {
            deadline = Clock::now() + std::chrono::seconds(owner.config.request_timeout_seconds);
            std::lock_guard<std::mutex> guard(owner.ownership);
            if (owner.stopping)
                throw HttpError(503, "server is stopping");
            if (owner.active >= owner.config.queue_limit)
                throw RequestOverloaded("HTTP request capacity reached");
            request.request_id = "speech-" + std::to_string(++owner.sequence);
            ++owner.active;
        }
        ~Job() {
            if (submitted && !terminal) {
                try {
                    owner.engine.cancel_request(request.request_id);
                } catch (const std::exception &error) {
                    TTS_LOG(TTS_LOG_WARN, "http", "cancel {}: {}", request.request_id,
                            error.what());
                }
            }
            std::lock_guard<std::mutex> guard(owner.ownership);
            --owner.active;
        }
        void check(const std::function<bool()> &disconnected) const {
            if (owner.stopping)
                throw HttpError(503, "server is stopping");
            if (disconnected())
                throw HttpError(499, "client disconnected");
            if (Clock::now() >= deadline)
                throw HttpError(504, "generation deadline exceeded");
        }
        GenerationFrame poll(int timeout, const std::function<bool()> &disconnected) {
            check(disconnected);
            GenerationFrame frame;
            try {
                frame = owner.engine.read_result(request.request_id, timeout);
            } catch (...) {
                terminal = true;
                throw;
            } // Native terminal errors consume the ID.
            if (frame.ready) {
                terminal = frame.done;
                stats = frame.stats;
                if (frame.cancelled)
                    throw HttpError(422, "generation cancelled");
            }
            return frame;
        }
        GenerationFrame next(const std::function<bool()> &disconnected) {
            for (;;) {
                auto frame = poll(100, disconnected);
                if (frame.ready)
                    return frame;
            }
        }
    };

    explicit Impl(Engine &native_engine, const Config &settings)
        : engine(native_engine), config(settings), audio(engine.audio_format()) {
        config.validate();
        if (config.model_name.empty())
            config.model_name = engine.model_info().model_type;
        http.new_task_queue = [this] {
            return new httplib::ThreadPool(2, config.threads, config.queue_limit);
        };
        http.set_payload_max_length(64 * 1024 * 1024);
        http.set_read_timeout(30);
        http.set_write_timeout(30);
        http.set_keep_alive_timeout(5);
        http.set_start_handler([this] {
            TTS_LOG(TTS_LOG_INFO, "http", "ready on http://{}:{}", config.host, config.port);
        });
        http.set_default_headers(
            {{"Cache-Control", "no-store"}, {"X-Content-Type-Options", "nosniff"}});
        http.set_error_handler([](const httplib::Request &, httplib::Response &response) {
            if (response.body.empty())
                json_reply(response, response.status,
                           error_json(response.status, httplib::status_message(response.status)));
        });
        http.set_exception_handler(
            [](const httplib::Request &, httplib::Response &response, std::exception_ptr error) {
                try {
                    std::rethrow_exception(error);
                } catch (const std::exception &failure) {
                    TTS_LOG(TTS_LOG_ERROR, "http", "{}", failure.what());
                    json_reply(response, 500, error_json(500, "internal server error"));
                }
            });
        http.Get("/health", [this](const httplib::Request &, httplib::Response &response) {
            json_reply(response, stopping ? 503 : 200,
                       {{"status", stopping ? "stopping" : "ready"},
                        {"engine", tts_version()},
                        {"sample_rate", audio.sample_rate},
                        {"max_batch", engine.batch_capacity()}});
        });
        const auto &info = engine.model_info();
        Json metadata{{"id", config.model_name},
                      {"model_type", info.model_type},
                      {"sample_rate", audio.sample_rate},
                      {"frame_rate", double(audio.sample_rate) / audio.samples_per_frame},
                      {"max_new_tokens", info.max_output_frames},
                      {"max_batch_texts", config.batch_max_items}};
        http.Get("/v1/model", [metadata](const httplib::Request &, httplib::Response &response) {
            json_reply(response, 200, metadata);
        });
        Json models{{"object", "list"},
                    {"data", Json::array({{{"id", config.model_name},
                                           {"object", "model"},
                                           {"created", 0},
                                           {"owned_by", "local"}}})}};
        http.Get("/v1/models", [models](const httplib::Request &, httplib::Response &response) {
            json_reply(response, 200, models);
        });
        http.Get(R"(/v1/audio/speech/([A-Za-z0-9._:-]+))",
                 [this](const httplib::Request &request, httplib::Response &response) {
                     Json result;
                     {
                         std::lock_guard<std::mutex> guard(ownership);
                         for (const auto &entry : outcomes)
                             if (entry.first == request.matches[1].str()) {
                                 result = entry.second;
                                 break;
                             }
                     }
                     if (result.is_null())
                         json_reply(response, 404, error_json(404, "speech outcome not found"));
                     else
                         json_reply(response, 200, result);
                 });
        for (const char *route : {"/v1/audio/speech", "/v1/audio/speech/batch"})
            http.Post(route, [this](const httplib::Request &request, httplib::Response &response) {
                try {
                    auto payload = parse_body(request);
                    if (request.path == "/v1/audio/speech/batch")
                        batch(request, response, payload);
                    else
                        speech(request, response, payload);
                } catch (...) {
                    auto error = api_error(std::current_exception());
                    json_reply(response, error["error"]["code"].get<int>(), error);
                }
            });
    }

    std::shared_ptr<const Reference> reference(const Json &item, const std::string &uploaded) {
        auto text = text_field(item, "ref_text", 65536);
        auto language = text_field(item, "prompt_language", 32);
        const auto encoded = item.value("ref_audio", Json());
        auto data = decode_reference(encoded);
        if (data.empty())
            data = uploaded;
        if (data.empty() != text.empty())
            throw std::invalid_argument("ref_audio and ref_text must be provided together");
        if (data.empty())
            return {};
        {
            std::lock_guard<std::mutex> guard(ownership);
            for (auto entry = references.begin(); entry != references.end(); ++entry) {
                if (entry->audio != data || entry->text != text || entry->language != language)
                    continue;
                auto result = entry->reference;
                references.splice(references.begin(), references, entry);
                return result;
            }
        }
        std::shared_ptr<const Reference> prepared;
        try {
            auto mono = read_audio(data.data(), data.size(), engine.max_reference_samples(),
                                   audio.sample_rate);
            prepared = engine.prepare_reference(mono, text, language);
        } catch (const RequestOverloaded &) {
            throw;
        } catch (const std::exception &error) {
            throw std::invalid_argument(error.what());
        }
        {
            std::lock_guard<std::mutex> guard(ownership);
            cached_bytes += data.size();
            references.push_front(
                {std::move(data), std::move(text), std::move(language), prepared});
            while (references.size() > 16 || cached_bytes > 64 * 1024 * 1024) {
                cached_bytes -= references.back().audio.size();
                references.pop_back();
            }
        }
        return prepared;
    }

    void audio_headers(httplib::Response &response) const {
        response.set_header("X-Channels", "1");
        response.set_header("X-Bit-Depth", "16");
        response.set_header("X-Sample-Rate", std::to_string(audio.sample_rate));
        response.set_header("X-Sample-Format", "s16le");
        response.set_header("X-Samples-Per-Frame", std::to_string(audio.samples_per_frame));
    }

    std::shared_ptr<Job> prepare(Json value, const std::string &uploaded,
                                 const std::function<bool()> &disconnected) {
        auto specification = parse_speech(value, engine.default_options());
        auto model = text_field(value, "model", 1024, config.model_name);
        if (model != config.model_name)
            throw HttpError(404, "model is not served: " + model);
        auto job = std::make_shared<Job>(*this);
        job->speech = std::move(specification);
        job->request.options = job->speech.options;
        job->speed = std::make_unique<AudioSpeed>(audio.sample_rate, job->speech.speed);
        job->check(disconnected);
        job->speech.input.reference = reference(value, uploaded);
        job->request.inputs.push_back(std::move(job->speech.input));
        return job;
    }
    void start(Job &job, const std::function<bool()> &disconnected) {
        job.check(disconnected);
        engine.add_request(job.request);
        job.submitted = true;
    }
    void record(const Job &job) {
        Json outcome{{"request_id", job.request.request_id},
                     {"finish_reason", finish_reason(job.stats)},
                     {"usage", nullptr}};
        std::lock_guard<std::mutex> guard(ownership);
        outcomes.emplace_back(job.request.request_id, std::move(outcome));
        if (outcomes.size() > 1024)
            outcomes.pop_front();
    }
    void identity_headers(const httplib::Request &request, httplib::Response &response,
                          const std::string &id) const {
        response.set_header("X-Request-Id", request.get_header_value("X-Request-Id", id.c_str()));
    }
    static void append(const GenerationFrame &frame, std::string &pcm) {
        for (const auto &chunk : frame.chunks) {
            if (chunk.first != 0)
                throw std::runtime_error("invalid engine output lane");
            pcm += pcm_bytes(chunk.second);
        }
    }
    void speech(const httplib::Request &request, httplib::Response &response,
                const Payload &payload) {
        auto job = prepare(payload.value, payload.uploaded_audio, request.is_connection_closed);
        start(*job, request.is_connection_closed);
        auto frame = job->next(request.is_connection_closed);
        identity_headers(request, response, job->request.request_id);
        audio_headers(response);
        if (job->speech.stream) {
            response.set_header("X-SGLang-Omni-Speech-Id", job->request.request_id);
            response.set_chunked_content_provider(
                job->speech.sse ? "text/event-stream" : "audio/pcm",
                [this, job, pending = std::move(frame)](size_t, httplib::DataSink &sink) mutable {
                    auto event = [&](const Json &value) {
                        auto bytes = "data: " + value.dump() + "\n\n";
                        return sink.write(bytes.data(), bytes.size());
                    };
                    try {
                        for (;;) {
                            job->check([&] { return !sink.is_writable(); });
                            std::string pcm;
                            append(pending, pcm);
                            pcm = job->speed->process(std::move(pcm), pending.done);
                            if (!pcm.empty()) {
                                bool written =
                                    job->speech.sse
                                        ? event({{"type", "speech.audio.delta"},
                                                 {"audio", httplib::detail::base64_encode(pcm)}})
                                        : sink.write(pcm.data(), pcm.size());
                                if (!written)
                                    return false;
                            }
                            if (pending.done) {
                                record(*job);
                                if (job->speech.sse &&
                                    !event({{"type", "speech.audio.done"},
                                            {"finish_reason", finish_reason(job->stats)},
                                            {"usage", nullptr}}))
                                    return false;
                                sink.done();
                                return true;
                            }
                            pending = job->next([&] { return !sink.is_writable(); });
                        }
                    } catch (...) {
                        auto error = api_error(std::current_exception());
                        if (job->speech.sse && sink.is_writable()) {
                            error["type"] = "error";
                            if (event(error)) {
                                sink.done();
                                return true;
                            }
                        }
                        return false;
                    }
                },
                [job](bool) {});
            return;
        }
        std::string pcm;
        for (;;) {
            append(frame, pcm);
            if (frame.done)
                break;
            frame = job->next(request.is_connection_closed);
        }
        response.set_header("X-Finish-Reason", finish_reason(job->stats));
        response.set_header("X-Engine-Time", std::to_string(job->stats.total_ms / 1000));
        response.set_header("Content-Disposition",
                            "attachment; filename=\"speech." + job->speech.format + "\"");
        pcm = job->speed->process(std::move(pcm), true);
        response.set_content(job->speech.format == "wav" ? wav_bytes(pcm, audio.sample_rate)
                                                         : std::move(pcm),
                             job->speech.format == "wav" ? "audio/wav" : "audio/pcm");
    }
    void batch(const httplib::Request &request, httplib::Response &response,
               const Payload &payload) {
        auto defaults = payload.value;
        auto items = defaults.value("items", Json());
        if (!items.is_array() || items.empty() || items.size() > config.batch_max_items)
            throw std::invalid_argument("items must contain 1.." +
                                        std::to_string(config.batch_max_items) + " entries");
        defaults.erase("items");
        auto envelope = defaults;
        envelope["input"] = "batch";
        parse_speech(envelope, engine.default_options());
        if ((defaults.contains("stream") &&
             (!defaults.at("stream").is_boolean() || defaults.at("stream").get<bool>())) ||
            text_field(defaults, "stream_format", 16, "audio") != "audio")
            throw std::invalid_argument("batch speech does not support streaming");
        std::string batch_id;
        {
            std::lock_guard<std::mutex> guard(ownership);
            batch_id = "speech-batch-" + std::to_string(++sequence);
        }
        std::vector<std::shared_ptr<Job>> jobs(items.size());
        std::vector<std::string> pcm(items.size());
        Json results = Json::array();
        for (size_t index = 0; index < items.size(); ++index)
            results.push_back({{"index", index}});
        auto failed = [&](size_t index, std::exception_ptr error) {
            results[index]["status"] = "error";
            results[index]["error"] = api_error(error).at("error");
            jobs[index].reset();
        };
        // Prepare all references before synthesis starts, then submit every valid item.
        for (size_t index = 0; index < items.size(); ++index) {
            try {
                if (!items[index].is_object())
                    throw std::invalid_argument("batch item must be an object");
                auto merged = defaults;
                merged.update(items[index]);
                jobs[index] = prepare(std::move(merged), payload.uploaded_audio,
                                      request.is_connection_closed);
                if (jobs[index]->speech.stream)
                    throw std::invalid_argument("batch speech does not support streaming");
            } catch (...) {
                failed(index, std::current_exception());
            }
        }
        size_t remaining = 0, succeeded = 0;
        for (size_t index = 0; index < jobs.size(); ++index) {
            if (!jobs[index])
                continue;
            try {
                start(*jobs[index], request.is_connection_closed);
                ++remaining;
            } catch (...) {
                failed(index, std::current_exception());
            }
        }
        auto consume = [&](size_t index, int timeout) {
            try {
                auto frame = jobs[index]->poll(timeout, request.is_connection_closed);
                if (!frame.ready)
                    return false;
                append(frame, pcm[index]);
                if (frame.done) {
                    const auto &job = *jobs[index];
                    auto adjusted = job.speed->process(std::move(pcm[index]), true);
                    auto bytes = job.speech.format == "wav" ? wav_bytes(adjusted, audio.sample_rate)
                                                            : std::move(adjusted);
                    results[index].update(
                        {{"status", "success"},
                         {"audio_data", httplib::detail::base64_encode(bytes)},
                         {"format", job.speech.format},
                         {"media_type", job.speech.format == "wav" ? "audio/wav" : "audio/pcm"},
                         {"finish_reason", finish_reason(job.stats)}});
                    jobs[index].reset();
                    --remaining;
                    ++succeeded;
                    pcm[index].clear();
                }
                return true;
            } catch (...) {
                failed(index, std::current_exception());
                --remaining;
                return true;
            }
        };
        // Drain every item fairly so independent generation never waits for the first item to
        // finish.
        while (remaining) {
            if (request.is_connection_closed() || stopping)
                throw HttpError(499, "client disconnected or server stopping");
            bool ready = false;
            for (size_t index = 0; index < jobs.size(); ++index)
                if (jobs[index])
                    ready = consume(index, 0) || ready;
            if (!ready)
                for (size_t index = 0; index < jobs.size(); ++index)
                    if (jobs[index]) {
                        consume(index, 10);
                        break;
                    }
        }
        identity_headers(request, response, batch_id);
        audio_headers(response);
        json_reply(response, 200,
                   {{"id", batch_id},
                    {"results", results},
                    {"total", items.size()},
                    {"succeeded", succeeded},
                    {"failed", items.size() - succeeded}});
    }
};

Server::Server(Engine &engine, const Config &config)
    : impl_(std::make_unique<Impl>(engine, config)) {}
Server::~Server() = default;
void Server::bind() {
    if (!impl_->http.bind_to_port(impl_->config.host, impl_->config.port))
        throw std::runtime_error("cannot bind HTTP server to " + impl_->config.host + ":" +
                                 std::to_string(impl_->config.port));
}
void Server::listen() {
    impl_->engine.configure_serving(impl_->config.queue_limit, impl_->config.stream_buffer_bytes);
    if (!impl_->http.listen_after_bind() && !impl_->stopping)
        throw std::runtime_error("HTTP listen failed");
}
void Server::stop() {
    impl_->stopping = true;
    impl_->http.stop();
}
} // namespace tts::http
