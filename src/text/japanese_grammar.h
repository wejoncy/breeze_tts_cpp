#pragma once
#include "io/json.h"
#include <deque>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace tts::text::japanese {
struct Node {
    std::string string, pos, pos_group1, pos_group2, pos_group3, ctype, cform,
                orig, read, pron, chain_rule;
    int acc = 0, mora_size = 0, chain_flag = -1;
    Node *prev = nullptr, *next = nullptr;
};
inline const char *get(const std::string &value) { return value.empty() ? "*" : value.c_str(); }
inline void assign(std::string &destination, const char *value) { destination = value ? value : ""; }
struct Rules {
    Json data;
    int max_mora;
    std::map<std::string, const char *, std::less<>> constants;
    std::map<std::string, std::vector<const char *>, std::less<>> tables;
    explicit Rules(const std::string &path);
    Rules(const Rules &) = delete;
    Rules &operator=(const Rules &) = delete;
    const char *constant(std::string_view name) const;
    const std::vector<const char *> &table(std::string_view name) const;
};
// Nodes remain owned by this request even when grammar rules unlink them.
// This gives stable neighboring pointers without manual allocation or frees.
struct Document {
    Document() = default;
    Document(const Document &) = delete;
    Document &operator=(const Document &) = delete;
    std::deque<Node> storage;
    Node *head = nullptr, *tail = nullptr;
    Node *create();
    void append(std::string_view feature);
    void load(Node *, std::string_view feature);
    Node *remove(Node *);
    void remove_silent();
};
void pronounce(Document &, const Rules &);
std::vector<std::string> prosody(const Document &, const Rules &);
} // namespace tts::text::japanese
