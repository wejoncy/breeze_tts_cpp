#include "audio_encoder.h"
#include "codec.h"
#include "model_config.h"
#include <filesystem>
#include <fstream>
#include <iostream>

// The development oracle generates alternate checkpoints and numerical inputs.
// This executable has no Python dependency and exercises the actual model loaders.
int main(int argc, char **argv) {
    if (argc != 4)
        throw std::invalid_argument("usage: test_audio_model MODEL INPUT_F32 OUTPUT_PREFIX");
    const std::string directory = argv[1], prefix = argv[3];
    auto config = tts::load_model_config(directory, 1, 256);
    tts::Backend gpu;
    tts::SafeTensors files;
    files.open_file(directory + "/audio_tokenizer/model.safetensors", "codec.");
    tts::Weights weights(gpu, std::move(files));
    tts::AudioEncoder encoder(weights, config.encoder);
    tts::Codec codec(weights, config.codec);
    weights.seal();
    std::ifstream input(std::filesystem::u8path(argv[2]), std::ios::binary | std::ios::ate);
    auto bytes = input.tellg();
    if (bytes <= 0 || bytes % sizeof(float))
        throw std::invalid_argument("invalid input samples");
    std::vector<float> samples(size_t(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(samples.data()), bytes);
    tts::DeviceArena arena(gpu.memory, encoder.workspace_bytes(samples.size()));
    auto codes = encoder.encode(gpu, arena, samples);
    gpu.sync();
    std::ofstream(std::filesystem::u8path(prefix + ".codes"), std::ios::binary)
        .write(reinterpret_cast<const char *>(codes.data()), codes.size() * sizeof(int));
    arena.reset();
    tts::WorkspacePlan plan;
    tts::CodecState planned(codec, 1, &plan);
    plan.allocate(size_t(config.codebooks) * sizeof(int));
    arena.reserve(plan.size);
    tts::CodecState state(codec, 1, &arena);
    state.reset(gpu);
    tts::Buffer frame(size_t(config.codebooks) * sizeof(int), &arena);
    std::vector<float> output(codes.size() / config.codebooks * config.samples_per_frame);
    for (size_t i = 0; i < codes.size() / config.codebooks; ++i) {
        gpu.upload(frame.data(), codes.data() + i * config.codebooks, frame.size());
        auto audio = codec.forward(gpu, state, frame.ptr<int>());
        gpu.download(output.data() + i * config.samples_per_frame, audio.data, audio.size());
    }
    std::ofstream(std::filesystem::u8path(prefix + ".audio"), std::ios::binary)
        .write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
    gpu.sync();
    std::cout << "encoded " << samples.size() << " samples; decoded " << output.size() << '\n';
}
