#include "safetensors.h"
#include "json.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace tts;
int main() {
    auto path = std::filesystem::temp_directory_path() /
                ("tts-reader-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    auto write = [&](const Json &header) {
        std::string json = header.dump();
        uint64_t size = json.size();
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(&size), 8);
        out.write(json.data(), json.size());
        float values[] = {1.f, 2.f};
        out.write(reinterpret_cast<const char *>(values), sizeof(values));
    };
    Json tensor = {{"dtype", "F32"}, {"shape", {1, 2}}, {"data_offsets", {0, 8}}};
    write(Json{{"a", tensor}});
    {
        SafeTensors reader;
        reader.open_file(path.u8string(), "model.");
        // A second mapping to the same file preserves zero-copy data lifetime.
        reader.open_file(path.u8string(), "other.");
        bool rejected = false;
        try {
            reader.open_file(path.u8string(), "model.");
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        assert(rejected);
        auto moved = std::move(reader);
        assert(moved.get("model.a").shape == std::vector<int>({1, 2}));
        float values[2];
        std::memcpy(values, moved.get("model.a").data, sizeof(values));
        assert(values[0] == 1.f && values[1] == 2.f);
    }
    Json invalid = tensor;
    invalid["data_offsets"] = {0, 128};
    write(Json{{"a", tensor}, {"z", invalid}});
    SafeTensors reader;
    bool rejected = false;
    try {
        reader.open_file(path.u8string());
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected && !reader.has("a") && !reader.has("z"));
    std::filesystem::remove(path);
    std::cout << "safetensors: mapped lifetime, namespace, duplicate and atomic failure passed\n";
}
