#include "models/breeze/codec.h"
#include "models/breeze/model_config.h"
#include <cassert>
#include <iostream>
// Keep GEMM geometry equal: compare dense histories with permuted borrowed
// histories and an inactive row. The dense fourth lane must not affect live rows.
int main(int argc, char **argv) {
    if (argc != 2)
        return 2;
    constexpr int batch = 3, padded_batch = batch + 1;
    auto config = tts::load_model_config(argv[1], batch, 256);
    tts::Backend gpu;
    tts::SafeTensors files;
    files.open_file(std::string(argv[1]) + "/audio_tokenizer/model.safetensors", "codec.");
    tts::Weights weights(gpu, std::move(files));
    tts::Codec codec(weights, config.codec);
    weights.seal();
    tts::CodecState dense(codec, padded_batch, &gpu.memory),
        scratch(codec, padded_batch, &gpu.memory);
    std::vector<std::unique_ptr<tts::CodecState>> state;
    for (int i = 0; i < batch; ++i) {
        state.push_back(std::make_unique<tts::CodecState>(codec, 1, &gpu.memory));
        state.back()->reset(gpu);
    }
    dense.reset(gpu);
    tts::Buffer codes(padded_batch * config.codebooks * sizeof(int), &gpu.memory),
        code_rows(padded_batch * sizeof(int *), &gpu.memory),
        caches(padded_batch * sizeof(tts::CacheSequence), &gpu.memory),
        history(padded_batch * dense.conv.size() * sizeof(tts::ConvHistory), &gpu.memory);
    std::vector<int> input(padded_batch * config.codebooks);
    std::vector<int16_t> expected(padded_batch * config.samples_per_frame),
        actual(padded_batch * config.samples_per_frame);
    int max_difference = 0;
    for (int frame = 0; frame < config.codec.transformer.window + 3; ++frame) {
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = (frame * 13 + i * 19) % config.codec.codebook_size;
        gpu.upload(codes.data(), input.data(), codes.size());
        codec.forward(gpu, dense, codes.ptr<int>());
        gpu.download(expected.data(), dense.pcm.data(), dense.pcm.size());
        std::vector<const int *> pointers(padded_batch, nullptr);
        std::vector<tts::CacheSequence> sequences(padded_batch);
        std::vector<tts::ConvHistory> histories(padded_batch * dense.conv.size());
        for (int row = 0; row < batch; ++row) {
            int lane = (row + frame) % batch;
            auto &s = *state[lane];
            pointers[row] = codes.ptr<int>() + lane * config.codebooks;
            sequences[row] = {s.transformer.keys.data(), s.transformer.values.data(),
                              s.transformer.keys.size() / config.codec.transformer.layers, nullptr,
                              s.position.ptr<int>()};
            for (size_t op = 0; op < s.conv.size(); ++op)
                histories[op * padded_batch + row] = {s.conv[op]->history.ptr<float>(),
                                                      s.position.ptr<int>()};
        }
        gpu.upload(code_rows.data(), pointers.data(), code_rows.size());
        gpu.upload(caches.data(), sequences.data(), caches.size());
        gpu.upload(history.data(), histories.data(), history.size());
        tts::CodecBatch mapped{code_rows.ptr<const int *>(), caches.ptr<tts::CacheSequence>(),
                               history.ptr<tts::ConvHistory>(), padded_batch};
        codec.forward(gpu, scratch, nullptr, padded_batch, &mapped);
        gpu.download(actual.data(), scratch.pcm.data(), scratch.pcm.size());
        for (size_t i = size_t(batch) * config.samples_per_frame; i < actual.size(); ++i)
            assert(actual[i] == 0);
        for (int row = 0; row < batch; ++row)
            for (int sample = 0; sample < config.samples_per_frame; ++sample) {
                int difference = std::abs(
                    int(actual[row * config.samples_per_frame + sample]) -
                    int(expected[((row + frame) % batch) * config.samples_per_frame + sample]));
                max_difference = std::max(max_difference, difference);
            }
    }
    std::cout << "permuted codec histories and inactive padding: max PCM difference "
              << max_difference << std::endl;
    assert(max_difference == 0);
}
