#pragma once
#include <blockingconcurrentqueue.h>
#include <new>
#include <utility>

namespace tts {
struct MailboxTraits : moodycamel::ConcurrentQueueDefaultTraits {
    static constexpr int MAX_SEMA_SPINS = 0;
};
// Message storage is lock-free; empty consumers park without spinning.
template <class T> using Mailbox = moodycamel::BlockingConcurrentQueue<T, MailboxTraits>;
template <class T, class U> void send(Mailbox<T> &queue, U &&value) {
    if (!queue.enqueue(std::forward<U>(value)))
        throw std::bad_alloc();
}
} // namespace tts
