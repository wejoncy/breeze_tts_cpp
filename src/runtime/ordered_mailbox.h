#pragma once
#include "runtime/mailbox.h"
#include <atomic>
#include <chrono>

namespace tts {
// Linearizable MPSC ingress. Producers publish complete nodes with one CAS;
// the sole consumer reverses each detached batch into FIFO order. Only queue
// internals synchronize; destruction requires all users to have stopped.
template <class T> class OrderedMailbox {
    struct Node {
        Node *next = nullptr;
        T value;
        template <class U> explicit Node(U &&v) : value(std::forward<U>(v)) {}
    };
    std::atomic<Node *> incoming_{nullptr};
    Node *local_ = nullptr;
    moodycamel::LightweightSemaphore wake_{0, 0};
    static void destroy(Node *node) noexcept {
        while (node) {
            Node *next = node->next;
            delete node;
            node = next;
        }
    }

  public:
    OrderedMailbox() = default;
    OrderedMailbox(const OrderedMailbox &) = delete;
    OrderedMailbox &operator=(const OrderedMailbox &) = delete;
    ~OrderedMailbox() {
        destroy(local_);
        destroy(incoming_.load(std::memory_order_relaxed));
    }
    template <class U> bool enqueue(U &&value) {
        auto *node = new Node(std::forward<U>(value));
        node->next = incoming_.load(std::memory_order_relaxed);
        // An old head may have been detached/freed, but producers only compare
        // its address. No producer dereferences a node owned by the consumer.
        while (!incoming_.compare_exchange_weak(node->next, node, std::memory_order_release,
                                                std::memory_order_relaxed)) {}
        wake_.signal();
        return true;
    }
    bool try_dequeue(T &value) {
        if (!local_) {
            auto *batch = incoming_.exchange(nullptr, std::memory_order_acquire);
            while (batch) {
                auto *next = batch->next;
                batch->next = local_;
                local_ = batch;
                batch = next;
            }
        }
        if (!local_) return false;
        value = std::move(local_->value);
        auto *next = local_->next;
        delete local_;
        local_ = next;
        // Notifications are wake events, not message permits: publication may
        // become visible before its producer signals this semaphore.
        wake_.tryWait();
        return true;
    }
    void wait_dequeue(T &value) {
        while (!try_dequeue(value)) wake_.wait();
    }
    bool wait_dequeue_timed(T &value, int64_t timeout_us) {
        if (timeout_us < 0) { wait_dequeue(value); return true; }
        const auto start = std::chrono::steady_clock::now();
        while (!try_dequeue(value)) {
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_us) return false;
            if (!wake_.wait(timeout_us - elapsed)) return try_dequeue(value);
        }
        return true;
    }
};
} // namespace tts
