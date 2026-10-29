#include "prepared_batch.h"
#include <cassert>
#include <iostream>
using namespace tts;
int main(int argc, char **argv) {
    assert(argc == 2);
    auto config = load_model_config(argv[1], 4, 2048);
    Tokenizer tokenizer(std::string(argv[1]) + "/tokenizer.json");
    auto ref = std::make_shared<Reference>();
    ref->text = "Hello.";
    ref->codebooks = config.codebooks;
    ref->codebook_size = config.codec.codebook_size;
    ref->codes.assign(2 * config.codebooks, 7);
    std::vector<Request> requests{{"你好。", "A calm voice.", "S0", ref},
                                  {"Welcome aboard.", "A clear voice.", "S1", {}}};
    GenerationOptions options;
    options.max_frames = 4;
    auto check = [&](const PreparedBatch &p) {
        assert(p.lanes == int(p.prompts.size()));
        assert(p.offsets.back() == p.prompt_rows);
        assert(p.text_rows == int(p.tokens.size()));
        assert(p.storage_rows >= p.prompt_rows);
        for (int lane = 0; lane < p.lanes; ++lane) {
            assert(p.offsets[lane + 1] - p.offsets[lane] == p.lengths[lane]);
            assert(p.capacities[lane] == p.lengths[lane] + options.max_frames);
            if (p.has_reference)
                for (int row = p.offsets[lane]; row < p.offsets[lane + 1]; ++row) {
                    assert(p.backbone_metadata[row] == lane);
                    assert(p.backbone_metadata[p.prompt_rows + row] == row - p.offsets[lane]);
                }
        }
        for (size_t lane = 0; lane < p.segments.size(); ++lane) {
            auto segment = p.segments[lane];
            for (int j = 0; j < segment.size; ++j) {
                assert(p.text_metadata[segment.start + j] == int(lane));
                assert(p.text_metadata[p.text_rows + segment.start + j] == j);
            }
        }
    };
    auto mixed = prepare_batch(tokenizer, config, 4, 2048, requests, options);
    check(mixed);
    assert(mixed.segments.size() == 3);
    assert(mixed.segments[1].destination == mixed.segments[0].size + 3); // two frames + EOS
    assert(mixed.prompts[0].target ==
           tokenizer.encode("[S0]<ins_bos>A calm voice.<ins_eos>你好。"));
    requests[1].reference = ref;
    options.dual_cfg = true;
    auto cfg = prepare_batch(tokenizer, config, 4, 2048, requests, options);
    check(cfg);
    assert(cfg.lanes == 6);
    for (int i = 0; i < 6; ++i)
        assert(bool(cfg.prompts[i].reference) == (i >= 2 && i < 4));
    assert(cfg.prompts[0].target == tokenizer.encode("[S0]你好。"));
    assert(cfg.prompts[2].target == cfg.prompts[0].target);
    assert(cfg.prompts[4].target == mixed.prompts[0].target);
    requests[0].reference.reset();
    bool rejected = false;
    try {
        prepare_batch(tokenizer, config, 4, 2048, requests, options);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    std::cout << "prepared batch: mixed reference layout, packed metadata, CFG branches, rejection "
                 "passed\n";
}
