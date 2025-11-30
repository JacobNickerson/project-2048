#include "shared_memory_structures.hpp"

void write_slot(ResponseCell *s, uint8_t move) {
    s->ready.store(0, std::memory_order_release);
    s->move = move;
    s->ready.store(1, std::memory_order_release);
}
uint8_t wait_read_slot(ResponseCell *s) {
    while (s->ready.load(std::memory_order_acquire) == 0) {}
    uint8_t val = s->move;
    s->ready.store(0);

    return s->move;
}