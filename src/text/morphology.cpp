#include "text/morphology.h"
#include <mecab.h>
#include <stdexcept>

namespace tts::text {
struct Morphology::Impl {
    std::unique_ptr<MeCab::Model> model;
    std::unique_ptr<MeCab::Tagger> tagger;
    Impl(const std::string &directory, const std::string &user_dictionary) {
        std::vector<std::string> arguments{"mecab", "-r", directory + "/dicrc", "-d", directory};
        if (!user_dictionary.empty()) {
            arguments.push_back("-u");
            arguments.push_back(user_dictionary);
        }
        std::vector<char *> values;
        for (auto &argument : arguments) values.push_back(argument.data());
        model.reset(MeCab::createModel(int(values.size()), values.data()));
        if (!model) throw std::runtime_error("MeCab dictionary initialization failed: " + std::string(MeCab::getLastError()));
        tagger.reset(model->createTagger());
        if (!tagger) throw std::runtime_error("MeCab tagger initialization failed: " + std::string(MeCab::getLastError()));
    }
};
Morphology::Morphology(const std::string &directory, const std::string &user_dictionary)
    : impl_(std::make_unique<Impl>(directory, user_dictionary)) {}
Morphology::~Morphology() = default;
std::vector<Morphology::Token> Morphology::analyze(std::string_view text) const {
    const auto *node = impl_->tagger->parseToNode(text.data(), text.size());
    if (!node) throw std::runtime_error("MeCab analysis failed: " + std::string(impl_->tagger->what()));
    std::vector<Token> result;
    for (; node; node = node->next)
        if (node->stat != MECAB_BOS_NODE && node->stat != MECAB_EOS_NODE)
            result.push_back({std::string(node->surface, node->length), node->feature});
    return result;
}
} // namespace tts::text
