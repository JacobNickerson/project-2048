#pragma once
#include <cstdint>
#include <iostream>

// branchless fast RNG, probably unnecessary but man is it fun
struct XorShift32 {
    uint32_t state;

    explicit XorShift32(uint32_t seed = 2463534242u) : state(seed) {}

    uint32_t next() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }

    uint32_t nextUInt(uint32_t n) {
        return next() % n;
    }
};
