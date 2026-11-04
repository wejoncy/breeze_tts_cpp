#pragma once
#include "runtime/sequence_state.h"
#include "core/host_tensor.h"
#include <map>
#include <string_view>

namespace tts {
class StatePool {
  public:
    struct TensorSpec {
        std::string name;
        size_t size;
        Type type = Type::I8;
        int columns = 1;
        bool host = false;
    };
    struct Layout {
        std::vector<TensorSpec> tensors;
        std::vector<std::pair<std::string, SequenceStateSpec>> histories;
    };
    class Row {
        std::map<std::string, Buffer, std::less<>> tensors_;
        std::map<std::string, HostTensor, std::less<>> host_;
        std::map<std::string, TensorView, std::less<>> views_;
        std::map<std::string, SequenceState, std::less<>> histories_;

      public:
        Row(const Layout &layout, BufferAllocator &storage) {
            for (const auto &field : layout.tensors) {
                const size_t row_bytes = size_t(field.columns) * bytes(field.type);
                if (!row_bytes || field.size % row_bytes || field.size / row_bytes > INT32_MAX)
                    throw std::invalid_argument("persistent tensor shape/type mismatch: " + field.name);
                const TensorShape shape{int(field.size / row_bytes), field.columns};
                TensorView view;
                if (field.host) {
                    auto tensor = host_.try_emplace(field.name, shape, field.type);
                    if (!tensor.second) throw std::invalid_argument("duplicate host state: " + field.name);
                    view = tensor.first->second.view();
                } else {
                    auto tensor = tensors_.try_emplace(field.name, field.size, &storage);
                    if (!tensor.second) throw std::invalid_argument("duplicate persistent tensor: " + field.name);
                    view = tensor.first->second.view(shape.rows, shape.cols, field.type);
                }
                if (!views_.emplace(field.name, view).second)
                    throw std::invalid_argument("duplicate persistent state: " + field.name);
            }
            for (const auto &field : layout.histories)
                if (!histories_.try_emplace(field.first, field.second, &storage).second)
                    throw std::invalid_argument("duplicate persistent history: " + field.first);
        }
        Buffer &operator[](std::string_view name) {
            auto field = tensors_.find(name);
            if (field == tensors_.end()) throw std::out_of_range("unknown persistent tensor");
            return field->second;
        }
        const Buffer &operator[](std::string_view name) const {
            auto field = tensors_.find(name);
            if (field == tensors_.end()) throw std::out_of_range("unknown persistent tensor");
            return field->second;
        }
        SequenceState &history(std::string_view name) {
            auto field = histories_.find(name);
            if (field == histories_.end()) throw std::out_of_range("unknown persistent history");
            return field->second;
        }
        const auto &fields() const { return tensors_; }
        const auto &views() const { return views_; }
        void reset_host() {
            for (auto &field : host_) {
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
        WorkspacePlan planning;
        Row measured(layout, planning);
        if (planning.size && count > SIZE_MAX / planning.size)
            throw std::overflow_error("persistent state capacity");
        storage_.reserve(count * planning.size);
        layout_ = layout;
        rows_.reserve(count);
        for (size_t row = 0; row < count; ++row)
            rows_.push_back(std::make_unique<Row>(layout, storage_));
    }
    const std::vector<std::unique_ptr<Row>> &rows() const { return rows_; }
    const SequenceStateSpec &history_spec(std::string_view name) const {
        for (const auto &field : layout_.histories)
            if (field.first == name) return field.second;
        throw std::out_of_range("unknown persistent history specification");
    }
};
} // namespace tts
