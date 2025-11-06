#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <iostream>

template<typename T>
class LockFreeQueue {
    static_assert(std::is_trivially_copyable<T>::value, "LockFreeQueue elements must be trivially copyable"); 
    public:
        LockFreeQueue(size_t cap) : 
            capacity(cap), mask(cap-1), head(0),
            tail(0) {
                assert((cap & (cap-1)) == 0); // force capacity to be a power of 2
            }

        size_t size() const {
            size_t h = head.load(std::memory_order_acquire);
            size_t t = tail.load(std::memory_order_acquire);
            return (t >= h) ? t-h : t-h+capacity;
        }
        bool push(T* buffer, T item) noexcept {
            size_t t = tail.load(std::memory_order_relaxed);
            size_t next = (t + 1) & mask;
            size_t h = head.load(std::memory_order_acquire);
            if (next == h) {
                // full
                return false;
            }
            // write data then publish
            buffer[t] = item;
            tail.store(next, std::memory_order_release);
            return true;
        }
        std::pair<T,bool> pop(T* buffer) noexcept {
            size_t h = head.load(std::memory_order_relaxed);
            size_t t = tail.load(std::memory_order_acquire);
            if (h == t) {
                // empty
                return std::make_pair(T(),false);
            }
            T val = buffer[h];
            size_t next = (h + 1) & mask;
            head.store(next, std::memory_order_release);
            return std::make_pair(val, true);
        }

    private:
        const size_t capacity;
        const size_t mask;

        std::atomic<size_t> head;
        std::atomic<size_t> tail;
};