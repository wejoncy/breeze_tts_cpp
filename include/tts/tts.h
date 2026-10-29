#ifndef TTS_CPP_H
#define TTS_CPP_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define TTS_ABI_VERSION 1
typedef struct tts_engine tts_engine;
typedef enum {
    TTS_LOG_TRACE, TTS_LOG_DEBUG, TTS_LOG_INFO, TTS_LOG_WARN, TTS_LOG_ERROR, TTS_LOG_OFF
} tts_log_level;
typedef struct {
    tts_log_level level;
    int64_t timestamp_us; /* Unix time, microseconds */
    uint64_t engine_id, request_id; /* zero outside an engine/request */
    const char *component, *message;
} tts_log_record;
/* Synchronous, serialized callbacks on the emitting thread. Strings are borrowed
   for this call only. Do not reenter engine/configuration APIs from the callback. */
typedef void (*tts_log_callback)(void *user, const tts_log_record *record);
typedef struct {
    tts_log_level level;
    const char *file_path; /* NULL = stderr; files rotate at 10 MiB, retaining 3 backups */
    int32_t json; /* JSON lines for stderr/file; callbacks receive structured records */
    tts_log_callback callback; /* replaces stderr/file output; file_path must be NULL */
    void *user;
} tts_log_config;
/* Process-wide for this library. NULL restores INFO/stderr/text. Successful
   replacement waits for old callbacks, after which old user data may be released. */
int tts_configure_logging(const tts_log_config *, char *error, size_t error_capacity);
typedef struct {
    uint32_t abi_version;
    int32_t max_frames; /* 1..min(2000, context - prompt length) */
    int32_t top_k;      /* 0 = full distribution, otherwise 1..2048 */
    float temperature;  /* 0 = greedy */
    float cfg_scale;    /* instruction guidance, > 0 */
    uint64_t seed;
    int32_t use_cuda_graph;
} tts_options;
typedef struct {
    const char *text;
    const char *instruction;
} tts_request;
typedef struct {
    double prefill_ms;
    double first_audio_ms;
    double total_ms;
    double audio_seconds;
    int32_t frames;
    int32_t truncated;
} tts_stats;
/* PCM pointer is valid during callback only; return nonzero to cancel the batch.
   callback runs on the calling thread. Do not reenter/destroy the engine there. */
typedef int (*tts_audio_callback)(void *user, int32_t batch_index, const int16_t *pcm,
                                  size_t samples);
tts_options tts_default_options(void);
tts_engine *tts_create(const char *model_dir, int32_t device, int32_t max_batch, int32_t context,
                       char *error, size_t error_capacity);
/* Quantization is "fp8" or "int8"; NULL selects unquantized BF16. Immutable per engine.
   Weights use per-output-channel scales; activations use dynamic per-token scales.
   FlashAttention/KV stay BF16, and the audio encoder/codec stay F32. */
tts_engine *tts_create_with_quantization(const char *model_dir, int32_t device, int32_t max_batch,
                                      int32_t context, const char *quantization, char *error,
                                      size_t error_capacity);
void tts_destroy(tts_engine *);
/* Engine-owned pool bytes. Driver/library-private allocations are not included.
   Synchronizes pending work; call between requests, never from an audio callback. */
typedef struct {
    uint64_t used_bytes, reserved_bytes, peak_used_bytes, peak_reserved_bytes;
} tts_memory_stats;
int tts_get_memory_stats(tts_engine *, tts_memory_stats *, char *error, size_t error_capacity);
/* An engine serializes calls; one invocation can contain a GPU batch of requests.
   Returns 0 on success, 1 on callback cancellation, -1 on error. */
int tts_generate(tts_engine *, const tts_request *, size_t count, const tts_options *,
                 tts_audio_callback, void *user, tts_stats *, char *error, size_t error_capacity);
const char *tts_version(void);

/* ABI 2 extends functionality without changing any ABI-1 layout/symbol. */
#define TTS_ABI_VERSION_2 2
typedef struct tts_reference tts_reference;
typedef struct {
    uint32_t abi_version;
    int32_t max_frames;
    int32_t top_k;
    float temperature;
    float cfg_scale;
    uint64_t seed;
    int32_t use_cuda_graph;
    float top_p;
    float repetition_penalty;
    int32_t do_sample;
    int32_t depth_top_k;
    float depth_temperature;
    float depth_top_p;
    int32_t depth_do_sample;
    int32_t dual_cfg;
    float cfg_scale_ref;
    float cfg_scale_ins;
} tts_options_v2;
typedef struct {
    const char *text;
    const char *instruction;
    const char *speaker; /* NULL = S0; empty string = no prefix */
    const tts_reference *reference;
} tts_request_v2;
tts_options_v2 tts_default_options_v2(void);
/* Decode/downmix/resample explicitly, then run the native encoder. The result
   owns CPU codes and transcript and may be reused across engines/requests. */
tts_reference *tts_reference_from_audio(tts_engine *, const void *encoded, size_t bytes,
                                        const char *transcript, char *error, size_t error_capacity);
tts_reference *tts_reference_from_pcm(tts_engine *, const float *interleaved, size_t frames,
                                      int32_t sample_rate, int32_t channels, const char *transcript,
                                      char *error, size_t error_capacity);
tts_reference *tts_reference_from_codes(const int32_t *codes, size_t frames, const char *transcript,
                                        char *error, size_t error_capacity);
size_t tts_reference_frames(const tts_reference *);
/* Returns frame count; out capacity is in int32 code elements. */
size_t tts_reference_codes(const tts_reference *, int32_t *out, size_t capacity);
void tts_reference_destroy(tts_reference *);
int tts_generate_v2(tts_engine *, const tts_request_v2 *, size_t count, const tts_options_v2 *,
                    tts_audio_callback, void *user, tts_stats *, char *error,
                    size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
