#include "support/log.h"
#include "io/json.h"
#include <cassert>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <vector>

int main() {
    std::vector<tts::Json> records;
    auto callback = [](void *user, const tts_log_record *record) {
        static_cast<std::vector<tts::Json> *>(user)->push_back({{"level", record->level},
                                                                {"engine", record->engine_id},
                                                                {"request", record->request_id},
                                                                {"message", record->message}});
        assert(record->timestamp_us > 0);
        TTS_LOG(TTS_LOG_ERROR, "recursive", "must be suppressed");
        char error[128];
        assert(tts_configure_logging(nullptr, error, sizeof(error)) == -1);
    };
    tts_log_config config{TTS_LOG_INFO, nullptr, 0, callback, &records};
    assert(tts_configure_logging(&config, nullptr, 0) == 0);
    int evaluated = 0;
    TTS_LOG(TTS_LOG_DEBUG, "test", "{}", ++evaluated);
    assert(evaluated == 0);
    {
        tts::log::Scope engine(7);
        {
            tts::log::Scope request(7, 9);
            TTS_LOG(TTS_LOG_INFO, "test", "你好 {}", 42);
        }
        TTS_LOG(TTS_LOG_WARN, "test", "restored");
    }
    assert(records.size() == 2 && records[0]["request"] == 9 && records[1]["request"] == 0);
    assert(records[0]["message"] == "你好 42");
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i)
        threads.emplace_back([i] {
            tts::log::Scope scope(100 + i, i + 1);
            for (int n = 0; n < 100; ++n)
                TTS_LOG(TTS_LOG_INFO, "thread", "{}", i);
        });
    for (auto &thread : threads)
        thread.join();
    assert(records.size() == 402);
    for (size_t i = 2; i < records.size(); ++i)
        assert(records[i]["engine"].get<int>() - 99 == records[i]["request"].get<int>());
    // Replacement cannot return while an old callback still borrows user data.
    std::atomic<bool> entered{false}, release{false}, replaced{false}, replacing{false};
    struct Blocking {
        std::atomic<bool> &entered, &release;
    } blocking{entered, release};
    config.callback = [](void *user, const tts_log_record *) {
        auto &b = *static_cast<Blocking *>(user);
        b.entered = true;
        while (!b.release)
            std::this_thread::yield();
    };
    config.user = &blocking;
    assert(tts_configure_logging(&config, nullptr, 0) == 0);
    std::thread emitter([] { TTS_LOG(TTS_LOG_INFO, "test", "blocking callback"); });
    while (!entered)
        std::this_thread::yield();
    std::thread replacement([&] {
        replacing = true;
        assert(tts_configure_logging(nullptr, nullptr, 0) == 0);
        replaced = true;
    });
    while (!replacing)
        std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!replaced);
    release = true;
    emitter.join();
    replacement.join();
    assert(replaced);
    auto path =
        std::filesystem::temp_directory_path() /
        ("tts-日志-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".jsonl");
    auto filename = path.u8string();
    config = {TTS_LOG_INFO, filename.c_str(), 1, nullptr, nullptr};
    assert(tts_configure_logging(&config, nullptr, 0) == 0);
    TTS_LOG(TTS_LOG_INFO, "json", "quotes=\" newline=\n 中文");
    assert(tts_configure_logging(nullptr, nullptr, 0) == 0); // flush and close
    std::ifstream file(path);
    std::string line, extra;
    std::getline(file, line);
    auto json = tts::Json::parse(line);
    assert(json["component"] == "json" && json["message"] == "quotes=\" newline=\n 中文");
    assert(!std::getline(file, extra));
    file.close();
    std::filesystem::remove(path);
    config.level = TTS_LOG_OFF;
    config.file_path = nullptr;
    assert(tts_configure_logging(&config, nullptr, 0) == 0);
    TTS_LOG(TTS_LOG_ERROR, "test", "{}", ++evaluated);
    assert(evaluated == 0);
}
