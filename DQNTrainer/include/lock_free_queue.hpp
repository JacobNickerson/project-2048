#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <optional>


template<typename T>
class LockFreeQueue {
    static_assert(std::is_trivially_copyable<T>::value, "LockFreeQueue elements must be trivially copyable"); 
    public:
        LockFreeQueue(size_t cap) : 
            capacity(cap), mask(cap-1) {
                assert((cap & (cap-1)) == 0); // force capacity to be a power of 2
                tail.store(0,std::memory_order_relaxed);
                head.store(0,std::memory_order_relaxed);
            }

        size_t size() const {
            size_t t = tail.load(std::memory_order_acquire);
            size_t h = head.load(std::memory_order_acquire);
            return (h >= t) ? h-t : h-t+capacity;
        }
        bool push(T* buffer, T item) noexcept {
            size_t h = head.load(std::memory_order_relaxed);
            size_t next = (h + 1) & mask;
            if (next == tail.load(std::memory_order_acquire)) {
                return false;
            }
            buffer[h] = item;
            head.store(next, std::memory_order_release);
            return true;
        }
        std::optional<T> pop(T* buffer) noexcept {
            size_t t = tail.load(std::memory_order_relaxed);
            if (t == head.load(std::memory_order_acquire)) {
                return std::nullopt;
            }
            T val = buffer[t];
            size_t next = (t + 1) & mask;
            tail.store(next, std::memory_order_release);
            return val;
        }

    private:
        const size_t capacity;
        const size_t mask;

        std::atomic<size_t> head;
        std::atomic<size_t> tail;
};