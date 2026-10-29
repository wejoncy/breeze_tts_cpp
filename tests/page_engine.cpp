#include "engine.h"
#include <cassert>
#include <iostream>

int main(int argc, char **argv) {
    std::string model = argc > 1 ? argv[1] : "..";
    using Audio = std::vector<std::vector<int16_t>>;
    std::vector<Audio> baseline;
    for (int page : {128, 64, 256}) {
        tts::Engine engine(model, 0, 3, 512, page);
        for (int scenario = 0; scenario < 3; ++scenario) {
            tts::GenerationOptions options;
            options.max_frames = 4;
            options.do_sample = options.depth_do_sample = false;
            options.cfg_scale = scenario == 1 ? 3 : 1;
            options.dual_cfg = scenario == 2;
            options.cfg_scale_ref = 2;
            options.cfg_scale_ins = 3;
            std::vector<tts::Request> requests;
            for (int i = 0; i < (scenario ? 3 : 1); ++i) {
                tts::Request request;
                for (int repeat = 0; repeat < 1 + i * 40; ++repeat)
                    request.text += "Welcome aboard. ";
                request.instruction = "A calm voice.";
                if (options.dual_cfg) {
                    auto ref = std::make_shared<tts::Reference>();
                    ref->text = "Hello.";
                    ref->codebooks = engine.audio_format().codebooks;
                    ref->codebook_size = engine.audio_format().codebook_size;
                    ref->codes.assign(3 * 16, 1);
                    request.reference = ref;
                }
                requests.push_back(std::move(request));
            }
            auto run = [&] {
                Audio audio(requests.size());
                tts::Stats stats;
                engine.generate(
                    requests, options,
                    [&](int i, const int16_t *pcm, size_t count) {
                        audio[i].insert(audio[i].end(), pcm, pcm + count);
                        return true;
                    },
                    stats);
                for (const auto &pcm : audio)
                    assert(pcm.size() == 4 * 1920);
                return audio;
            };
            Audio graph = run();
            options.use_cuda_graph = false;
            assert(graph == run());
            if (page == 128)
                baseline.push_back(graph);
            else
                assert(graph == baseline[scenario]);
        }
        std::cout << "page size " << page << ": graph/eager, ragged CFG and dual CFG agree\n";
    }
}
