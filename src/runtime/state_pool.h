#pragma once
#include "runtime/sequence_state.h"
#include "core/tensor_storage.h"
#include <cstring>
#include <map>
#include <string_view>

namespace tts {
class StatePool {
  public:
    struct TensorSpec {
        std::string name;
        TensorShape shape;
        Type type;
        bool host = false;
    };
    struct Layout {
        std::vector<TensorSpec> tensors;
        std::vector<std::pair<std::string, SequenceStateSpec>> histories;
    };
    class Row {
        std::map<std::string, Tensor, std::less<>> tensors_;
        std::map<std::string, SequenceState, std::less<>> histories_;

      public:
        Row(const Layout &layout) {
            for (const auto &field : layout.tensors) {
                if (!field.shape.cols || field.shape.numel() > SIZE_MAX / bytes(field.type))
                    throw std::invalid_argument("persistent tensor shape/type mismatch: " + field.name);
                auto tensor = field.host ? Tensor::zeros(field.shape, field.type)
                    : Tensor(Buffer(field.shape.numel() * bytes(field.type)), field.shape, field.type);
                if (!tensors_.try_emplace(field.name, std::move(tensor)).second)
                    throw std::invalid_argument("duplicate persistent state: " + field.name);
            }
            for (const auto &field : layout.histories)
                if (!histories_.try_emplace(field.first, field.second).second)
                    throw std::invalid_argument("duplicate persistent history: " + field.first);
        }
        TensorView operator[](std::string_view name) const {
            auto field = tensors_.find(name);
            if (field == tensors_.end() || field->second.device().is_cpu()) throw std::out_of_range("unknown persistent tensor");
            return field->second.view();
        }
        SequenceState &history(std::string_view name) {
            auto field = histories_.find(name);
            if (field == histories_.end()) throw std::out_of_range("unknown persistent history");
            return field->second;
        }
        const auto &fields() const { return tensors_; }
        TensorView view(std::string_view name) const {
            auto field = tensors_.find(name);
            if (field == tensors_.end()) throw std::out_of_range("unknown persistent state");
            return field->second.view();
        }
        void reset_host() {
            for (const auto &field : tensors_) if (field.second.device().is_cpu()) {
                const auto tensor = field.second.view();
                if (tensor.size()) std::memset(tensor.data, 0, tensor.size());
            }
        }
    };

  private:
    DeviceArena storage_;
    std::vector<std::unique_ptr<Row>> rows_;
    Layout layout_;

  public:
    explicit StatePool(DeviceContext &memory) : storage_(memory) {}
    void prepare(const Layout &layout, size_t count) {
        if (!rows_.empty() || !count)
            throw std::logic_error("invalid persistent state initialization");
        const size_t row_bytes = Tensor::plan([&] { Row measured(layout); });
        if (row_bytes && count > SIZE_MAX / row_bytes)
            throw std::overflow_error("persistent state capacity");
        storage_.reserve(count * row_bytes);
        Tensor::Scope scope(storage_);
        layout_ = layout;
        rows_.reserve(count);
        for (size_t row = 0; row < count; ++row)
            rows_.push_back(std::make_unique<Row>(layout));
    }
    const std::vector<std::unique_ptr<Row>> &rows() const { return rows_; }
    const Layout &layout() const { return layout_; }
    const SequenceStateSpec &history_spec(std::string_view name) const {
        for (const auto &field : layout_.histories)
            if (field.first == name) return field.second;
        throw std::out_of_range("unknown persistent history specification");
    }
};
} // namespace tts
