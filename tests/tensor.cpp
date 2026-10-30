#include "core/tensor.h"
#include <cassert>
#include <utility>
#include <iostream>
using namespace tts;
template <class F> void rejects(F f) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception &) {
        caught = true;
    }
    assert(caught);
}
int main() {
    float data[24]{};
    TensorView v{data, 6, 4, Type::F32};
    auto odd = v.every_rows(1, 2, 3);
    assert(odd.ptr<float>() == data + 4 && odd.ld() == 8 && odd.span_bytes() == 80);
    assert(!odd.dense() && odd.size() == 48);
    assert(odd.slice(2).ptr<float>() == data + 20);
    assert(v.columns(1, 2).slice(3).ptr<float>() == data + 13);
    assert(v.reshape({3, 8}).data == data);
    rejects([&] { odd.reshape({3, 4}); });
    rejects([&] { v.every_rows(1, 2, 4); });
    rejects([&] { v.columns(3, 2); });
    rejects([&] { v.slice(6); });
    rejects([&] { TensorShape(-1, 4); });
    rejects([&] { TensorView(data, 2, 4, Type::F32, 3); });
    std::cout << "Tensor metadata, strided bounds, and no-copy reshape: PASS\n";
}
