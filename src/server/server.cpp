#include "server/server.h"
#include "server/protocol.h"
#include "web.h"
#include "tts/engine.h"
#include "io/audio_io.h"
#include "io/audio_speed.h"
#include "support/log.h"
#include "support/profiler.h"
#include <httplib.h>
#include <atomic>
#include <chrono>
#include <list>
#include <mutex>
#include <deque>
#include <unordered_map>

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
    // Every HTTP worker uses this same instance; Engine owns both queue directions.
    Engine &engine;
    Config config;
    AudioFormat audio;
    httplib::Server http;
    std::atomic<bool> stopping{false};
    // HTTP ownership only. Never hold this mutex across an Engine call or socket IO.
    std::mutex ownership;
    struct Job;
    std::unordered_map<std::string, Job *> connections;
    size_t active = 0;
    uint64_t sequence = 0;
    struct CachedReference {
        std::string audio, text, language;
        std::shared_ptr<const Reference> reference;
    };
    std::list<CachedReference> references;
    size_t cached_bytes = 0;

    struct Job {
        Impl &owner;
        Request request;
        Clock::time_point deadline;
        bool submitted = false, terminal = false, output_failed = false;
        std::deque<GenerationFrame> pending;
        size_t pending_bytes = 0;
        Speech speech;
        Stats stats;
        std::string pcm;
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
            owner.connections.emplace(request.request_id, this);
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
            owner.connections.erase(request.request_id);
            --owner.active;
        }
        template <class Disconnected> void check(const Disconnected &disconnected) const {
            if (owner.stopping)
                throw HttpError(503, "server is stopping");
            if (disconnected())
                throw HttpError(499, "client disconnected");
            if (Clock::now() >= deadline)
                throw HttpError(504, "generation deadline exceeded");
        }
        template <class Disconnected> bool poll(int timeout, const Disconnected &disconnected) {
            check(disconnected);
            GenerationFrame frame;
            if (!owner.receive(*this, frame, timeout)) return false;
            if (frame.error) { terminal = true; std::rethrow_exception(frame.error); }
            terminal = frame.done;
            stats = frame.stats;
            if (frame.cancelled)
                throw HttpError(422, "generation cancelled");
            std::string chunk;
            for (const auto &[lane, samples] : frame.chunks) {
                if (lane != 0)
                    throw std::runtime_error("invalid engine output lane");
                chunk += pcm_bytes(samples);
            }
            pcm += speed->process(std::move(chunk), terminal);
            return true;
        }
        template <class Disconnected> void next(const Disconnected &disconnected) {
            while (!poll(100, disconnected)) {
            }
        }
        std::string audio() {
            return speech.format == "wav" ? wav_bytes(pcm, owner.audio.sample_rate)
                                          : std::move(pcm);
        }
        const char *media_type() const {
            return speech.format == "wav" ? "audio/wav" : "audio/pcm";
        }
    };

    bool receive(Job &connection, GenerationFrame &frame, int timeout_ms) {
        auto &queue = engine.output_queue();
        std::unique_lock<std::mutex> queue_lock(queue.mutex);
        auto take = [&] {
            if (connection.pending.empty()) return false;
            frame = std::move(connection.pending.front());
            if (!frame.chunks.empty()) mark("tts.http.dequeue", frame_trace_id(frame.request_id, uint32_t(frame.stats.frames)));
            connection.pending.pop_front();
            for (const auto &chunk : frame.chunks) connection.pending_bytes -= chunk.second.size() * sizeof(int16_t);
            return true;
        };
        {
            std::lock_guard<std::mutex> guard(ownership);
            if (take()) return true;
        }
        if (queue.frames.empty() && !queue.closed && timeout_ms)
            queue.available.wait_for(queue_lock, std::chrono::milliseconds(timeout_ms));
        std::vector<std::string> cancelled;
        bool dispatched = !queue.frames.empty(), received = false;
        {
            std::lock_guard<std::mutex> guard(ownership);
            while (!queue.frames.empty()) {
                auto output = std::move(queue.frames.front());
                queue.frames.pop_front();
                size_t bytes = 0;
                for (const auto &chunk : output.chunks) bytes += chunk.second.size() * sizeof(int16_t);
                queue.bytes -= bytes;
                auto found = connections.find(output.request_id);
                if (found == connections.end() || found->second->output_failed) continue;
                auto &target = *found->second;
                if (bytes > config.stream_buffer_bytes - target.pending_bytes) {
                    target.pending.clear(); target.pending_bytes = 0; target.output_failed = true;
                    output.chunks.clear(); output.done = true;
                    output.error = std::make_exception_ptr(std::runtime_error("HTTP audio consumer exceeded its buffer limit"));
                    cancelled.push_back(output.request_id);
                } else target.pending_bytes += bytes;
                target.pending.push_back(std::move(output));
            }
            received = take();
        }
        const bool closed = queue.closed;
        queue_lock.unlock();
        if (dispatched) queue.available.notify_all();
        for (const auto &id : cancelled) engine.cancel_request(id);
        if (!received && closed) throw HttpError(503, "engine is shut down");
        return received;
    }

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
        mount_web_ui(http, config.voice_presets_file);
        http.Get("/health", [this](const httplib::Request &, httplib::Response &response) {
            json_reply(response, stopping ? 503 : 200,
                       {{"status", stopping ? "stopping" : "ready"},
                        {"engine", tts_version()},
                        {"sample_rate", audio.sample_rate},
                        {"max_batch", engine.batch_capacity()}});
        });
        const auto &info = engine.model_info();
        auto metadata = [&](const char *route, Json value) {
            http.Get(route, [value = std::move(value)](const httplib::Request &,
                                                       httplib::Response &response) {
                json_reply(response, 200, value);
            });
        };
        metadata("/v1/model", {{"id", config.model_name},
                               {"default_seed", std::to_string(engine.default_options().seed)},
                               {"model_type", info.model_type},
                               {"sample_rate", audio.sample_rate},
                               {"frame_rate", double(audio.sample_rate) / audio.samples_per_frame},
                               {"max_new_tokens", info.max_output_frames},
                               {"max_batch_texts", config.batch_max_items}});
        metadata("/v1/models", {{"object", "list"},
                                {"data", Json::array({{{"id", config.model_name},
                                                       {"object", "model"},
                                                       {"created", 0},
                                                       {"owned_by", "local"}}})}});
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

    template <class Disconnected> std::shared_ptr<Job> prepare(Json value, const std::string &uploaded,
                                 const Disconnected &disconnected) {
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
    template <class Disconnected> void start(Job &job, const Disconnected &disconnected) {
        job.check(disconnected);
        engine.add_request(job.request);
        job.submitted = true;
    }
    void identity_headers(const httplib::Request &request, httplib::Response &response,
                          const std::string &id) const {
        response.set_header("X-Request-Id", request.get_header_value("X-Request-Id", id.c_str()));
    }
    void speech(const httplib::Request &request, httplib::Response &response,
                const Payload &payload) {
        auto job = prepare(payload.value, payload.uploaded_audio, request.is_connection_closed);
        start(*job, request.is_connection_closed);
        job->next(request.is_connection_closed);
        identity_headers(request, response, job->request.request_id);
        audio_headers(response);
        if (job->speech.stream) {
            response.set_header("X-SGLang-Omni-Speech-Id", job->request.request_id);
            response.set_chunked_content_provider(
                job->speech.sse ? "text/event-stream" : "audio/pcm",
                [this, job](size_t, httplib::DataSink &sink) {
                    auto event = [&](const Json &value) {
                        auto bytes = "data: " + value.dump() + "\n\n";
                        return sink.write(bytes.data(), bytes.size());
                    };
                    try {
                        for (;;) {
                            job->check([&] { return !sink.is_writable(); });
                            if (!job->pcm.empty()) {
                                bool written =
                                    job->speech.sse
                                        ? event(
                                              {{"type", "speech.audio.delta"},
                                               {"audio", httplib::detail::base64_encode(job->pcm)}})
                                        : sink.write(job->pcm.data(), job->pcm.size());
                                if (!written)
                                    return false;
                                mark("tts.http.sent", frame_trace_id(job->request.request_id, uint32_t(job->stats.frames)));
                            }
                            job->pcm.clear();
                            if (job->terminal) {
                                if (job->speech.sse &&
                                    !event({{"type", "speech.audio.done"},
                                            {"finish_reason", finish_reason(job->stats)},
                                            {"usage", nullptr}}))
                                    return false;
                                sink.done();
                                return true;
                            }
                            job->next([&] { return !sink.is_writable(); });
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
        while (!job->terminal)
            job->next(request.is_connection_closed);
        response.set_header("X-Finish-Reason", finish_reason(job->stats));
        response.set_header("X-Engine-Time", std::to_string(job->stats.total_ms / 1000));
        response.set_header("Content-Disposition",
                            "attachment; filename=\"speech." + job->speech.format + "\"");
        response.set_content(job->audio(), job->media_type());
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
        if (parse_speech(envelope, engine.default_options()).stream)
            throw std::invalid_argument("batch speech does not support streaming");
        std::string batch_id;
        {
            std::lock_guard<std::mutex> guard(ownership);
            batch_id = "speech-batch-" + std::to_string(++sequence);
        }
        std::vector<std::shared_ptr<Job>> jobs(items.size());
        Json results = Json::array();
        for (size_t index = 0; index < items.size(); ++index)
            results.push_back({{"index", index}});
        size_t remaining = items.size(), succeeded = 0;
        auto attempt = [&](size_t index, auto work) {
            try {
                work();
            } catch (...) {
                results[index]["status"] = "error";
                results[index]["error"] = api_error(std::current_exception()).at("error");
                jobs[index].reset();
                --remaining;
            }
        };
        // Prepare references before synthesis starts, then submit every valid item.
        for (size_t index = 0; index < items.size(); ++index)
            attempt(index, [&] {
                if (!items[index].is_object())
                    throw std::invalid_argument("batch item must be an object");
                auto merged = defaults;
                merged.update(items[index]);
                jobs[index] = prepare(std::move(merged), payload.uploaded_audio,
                                      request.is_connection_closed);
                if (jobs[index]->speech.stream)
                    throw std::invalid_argument("batch speech does not support streaming");
            });
        for (size_t index = 0; index < jobs.size(); ++index)
            if (jobs[index])
                attempt(index, [&] { start(*jobs[index], request.is_connection_closed); });
        auto consume = [&](size_t index, int timeout) {
            bool ready = true;
            attempt(index, [&] {
                auto &job = *jobs[index];
                ready = job.poll(timeout, request.is_connection_closed);
                if (!ready || !job.terminal)
                    return;
                results[index].update({{"status", "success"},
                                       {"audio_data", httplib::detail::base64_encode(job.audio())},
                                       {"format", job.speech.format},
                                       {"media_type", job.media_type()},
                                       {"finish_reason", finish_reason(job.stats)}});
                jobs[index].reset();
                --remaining;
                ++succeeded;
            });
            return ready;
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
    if (impl_->config.stream_buffer_bytes > SIZE_MAX / impl_->config.queue_limit)
        throw std::invalid_argument("HTTP output capacity overflow");
    impl_->engine.configure_serving(impl_->config.queue_limit,
        impl_->config.stream_buffer_bytes * impl_->config.queue_limit);
    if (!impl_->http.listen_after_bind() && !impl_->stopping)
        throw std::runtime_error("HTTP listen failed");
}
void Server::stop() {
    impl_->stopping = true;
    impl_->http.stop();
}
} // namespace tts::http
