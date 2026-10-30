#include "tts/tts.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// The standalone C ABI is tested as a C client, independently of Python handles.
int main(int argc, char **argv) {
    assert(argc == 2);
    char error[4096]{};
    std::unique_ptr<tts_engine, decltype(&tts_destroy)> engine(
        tts_create(argv[1], 0, 1, 2048, nullptr, error, sizeof(error)), tts_destroy);
    assert(engine);
    tts_memory_stats memory{};
    assert(tts_get_memory_stats(engine.get(), &memory, error, sizeof(error)) == -1);
    assert(std::strstr(error, "run_async"));
    assert(tts_run_async(engine.get(), error, sizeof(error)) == 0);
    assert(tts_run_async(engine.get(), error, sizeof(error)) == 0);
    tts_audio_format format{};
    assert(tts_get_audio_format(engine.get(), &format, error, sizeof(error)) == 0);
    assert(format.sample_rate > 0 && format.samples_per_frame > 0 && format.codebooks > 1);
    std::vector<int32_t> codes(size_t(format.codebooks) * 2, 0), copied(codes.size());
    std::unique_ptr<tts_reference, decltype(&tts_reference_destroy)> reference(
        tts_reference_from_codes(codes.data(), 2, format.codebooks, format.codebook_size,
                                 "Reference", error, sizeof(error)),
        tts_reference_destroy);
    assert(reference && tts_reference_frames(reference.get()) == 2);
    assert(tts_reference_codes(reference.get(), copied.data(), copied.size()) == 2 &&
           codes == copied);
    codes[0] = format.codebook_size;
    assert(!tts_reference_from_codes(codes.data(), 2, format.codebooks, format.codebook_size,
                                     "Reference", error, sizeof(error)));
    auto receive = [](void *user, int32_t, const int16_t *pcm, size_t count) -> int {
        auto &out = *static_cast<std::vector<int16_t> *>(user);
        out.insert(out.end(), pcm, pcm + count);
        return 0;
    };
    std::vector<int16_t> first, second;
    tts_stats stats{};
    tts_request request{"Hello.", "", "S0", nullptr};
    auto options = tts_default_options();
    options.max_frames = 3;
    options.temperature = options.depth_temperature = 0;
    assert(tts_generate(engine.get(), &request, 1, &options, receive, &first, &stats, error,
                        sizeof(error)) == 0);
    assert(!first.empty());
    request.instruction = "Calm.";
    request.reference = reference.get();
    options.dual_cfg = 1;
    assert(tts_generate(engine.get(), &request, 1, &options, receive, &second, &stats, error,
                        sizeof(error)) == 0);
    assert(!second.empty());
    options.dual_cfg = 2;
    assert(tts_generate(engine.get(), &request, 1, &options, receive, &second, &stats, error,
                        sizeof(error)) == -1);
    assert(std::strstr(error, "boolean"));
    options.abi_version = TTS_ABI_VERSION - 1;
    assert(tts_generate(engine.get(), &request, 1, &options, receive, &second, &stats, error,
                        sizeof(error)) == -1);
    assert(std::strstr(error, "ABI"));
    assert(tts_shutdown(engine.get(), error, sizeof(error)) == 0);
    assert(tts_shutdown(engine.get(), error, sizeof(error)) == 0);
    assert(tts_run_async(engine.get(), error, sizeof(error)) == -1);
}
