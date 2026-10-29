#include "tts/tts.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

// The standalone C ABI is tested as a C client, independently of Python handles.
int main(int argc, char **argv) {
    assert(argc == 2);
    char error[4096]{};
    std::unique_ptr<tts_engine, decltype(&tts_destroy)> engine(
        tts_create(argv[1], 0, 1, 2048, error, sizeof(error)), tts_destroy);
    assert(engine);
    auto receive = [](void *user, int32_t, const int16_t *pcm, size_t count) -> int {
        auto &out = *static_cast<std::vector<int16_t> *>(user);
        out.insert(out.end(), pcm, pcm + count);
        return 0;
    };
    std::vector<int16_t> first, second;
    tts_stats stats{};
    tts_request request{"Hello.", ""};
    auto options = tts_default_options();
    options.max_frames = 3;
    options.temperature = 0;
    assert(tts_generate(engine.get(), &request, 1, &options, receive, &first, &stats, error,
                        sizeof(error)) == 0);
    auto options_v2 = tts_default_options_v2();
    options_v2.max_frames = options.max_frames;
    options_v2.temperature = options_v2.depth_temperature = 0;
    options_v2.repetition_penalty = 1;
    tts_request_v2 request_v2{"Hello.", "", "S0", nullptr};
    assert(tts_generate_v2(engine.get(), &request_v2, 1, &options_v2, receive, &second, &stats,
                           error, sizeof(error)) == 0);
    assert(!first.empty() && first == second);
}
