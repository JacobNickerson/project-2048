#pragma once

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <optional>
#include <vector>

#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

namespace bip = boost::interprocess;


// Queue that uses locks. Shame.
// To avoid excessive locking of the mutex, it is assumed it will never be full
template<typename T>
class LockQueue {
    static_assert(std::is_trivially_copyable<T>::value, "LockFreeQueue elements must be trivially copyable"); 
    public:
        LockQueue(size_t cap) : 
            capacity(cap), mask(cap-1) {
                assert((cap & (cap-1)) == 0); // force capacity to be a power of 2
            }

        size_t size() const {
            size_t t = tail;
            size_t h = head;
            return (h >= t) ? h-t : h-t+capacity;
        }
        bool push(T* buffer, T item) noexcept {
            auto next = (head + 1) & mask; 
            bip::scoped_lock<bip::interprocess_mutex> lock(mutex);
            buffer[head] = item;
            head = next;
            cond.notify_all();
            return true;
        }
        std::optional<T> pop(T* buffer) noexcept {
            if (tail == head) { return std::nullopt; }
            auto next = (tail + 1) & mask; 
            auto val = buffer[tail];
            tail = next;
            return val;
        }
        std::vector<T> popBatch(T* buffer) noexcept {
            size_t h = head;
            if (tail == h) {
                return std::vector<T>();
            }
            static constexpr auto size = sizeof(T);
            auto read_size = (tail < h) ? (h-tail) : (h-tail+capacity);
            std::vector<T> data;
            data.reserve(read_size);
            for (;tail != h; tail = (tail+1)&mask) {
                data.emplace_back(buffer[tail]);
            }
            return data;
        }


    private:
        const size_t capacity;
        const size_t mask;

        volatile size_t head=0;
        size_t tail=0;

        bip::interprocess_mutex mutex;
        bip::interprocess_condition cond;
};