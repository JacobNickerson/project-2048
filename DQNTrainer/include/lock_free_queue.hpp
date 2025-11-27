#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <optional>
#include <vector>


// Generic Lock-Free queue intended to be used for MPSC
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

            while (true) {
                size_t next = (h + 1) & mask;

                // check for full queue
                size_t t = tail.load(std::memory_order_acquire);
                if (next == t) {
                    return false; // queue full
                }

                // try to claim the slot
                if (head.compare_exchange_weak(
                        h, next,
                        std::memory_order_release,  // on success
                        std::memory_order_relaxed   // on failure
                    )) {
                    // successfully reserved the slot at 'h'
                    buffer[h] = item; 
                    return true;
                }

                // CAS failed → h updated to latest value, retry
            }
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
        std::optional<std::vector<T>> pop_all(T* buffer) noexcept {
            size_t t = tail.load(std::memory_order_relaxed);
            size_t h = head.load(std::memory_order_acquire);
            if (t == h) {
                return std::nullopt;
            }
            std::vector<T> data;
            constexpr auto size = sizeof(T);
            if (t < h) {
                auto read_size = h-t;
                data.resize(read_size); 
                memcpy(data.data(),buffer+t,read_size*size);
            } else {
               auto read_1 = capacity-t; 
               auto read_2 = h;
               data.resize(read_1+read_2);
               memcpy(data.data(),buffer+t,read_1*size);
               memcpy(data.data()+read_1,buffer,read_2*size);
            }
            tail.store(head, std::memory_order_release);
            return data;
        }


    private:
        const size_t capacity;
        const size_t mask;

        std::atomic<size_t> head;
        std::atomic<size_t> tail;
};