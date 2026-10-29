#include "log.h"

int run_cli(int argc, const char *const *argv);

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <cstdio>
#include <vector>

int wmain(int argc, wchar_t **argv) {
    try {
        if (_setmode(_fileno(stdout), _O_BINARY) == -1)
            throw std::runtime_error("cannot set binary stdout");
        std::vector<std::string> arguments;
        arguments.reserve(argc);
        for (int i = 0; i < argc; ++i) {
            int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, nullptr, 0,
                                           nullptr, nullptr);
            if (!size)
                throw std::runtime_error("invalid Unicode command line");
            std::string text(size, '\0');
            if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1, text.data(), size,
                                     nullptr, nullptr))
                throw std::runtime_error("cannot convert command line to UTF-8");
            text.pop_back();
            arguments.push_back(std::move(text));
        }
        std::vector<const char *> pointers;
        for (const auto &text : arguments)
            pointers.push_back(text.c_str());
        return run_cli(argc, pointers.data());
    } catch (const std::exception &error) {
        TTS_LOG(TTS_LOG_ERROR, "cli", "{}", error.what());
        return 1;
    }
}
#else
int main(int argc, char **argv) {
    return run_cli(argc, argv);
}
#endif
