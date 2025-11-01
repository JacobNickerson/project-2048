#pragma once

#include <cstdint>
#include <array>

#include <iostream>
#include <bitset>

struct RowEntry {
    uint16_t result;
    uint32_t score; 
};

// Each tile can have 16 different values, 4 tiles per row = 65536 rows
constexpr uint32_t MOVE_COUNT = (16*16*16*16); 

std::array<uint8_t, 4> unpackRow(uint16_t row);
uint16_t packRow(std::array<uint8_t,4> row);

uint16_t reverseRow(uint16_t row);
std::array<uint16_t,4> transposeBoard(std::array<uint16_t,4> board);

constexpr uint16_t rowShiftLeft(uint16_t row) {
    // this implementation is disgusting
    uint16_t r[4] = {0};
    r[0] = (row & 0xF000) >> 12;
    r[1] = (row & 0x0F00) >> 8 ;
    r[2] = (row & 0x00F0) >> 4 ;
    r[3] = (row & 0x000F) >> 0 ;
    // Compress
    for (int i{1}; i < 4; ++i) {
        int L = i-1;
        while (L >= 0 && r[L] == 0) {
            --L;
        }
        std::swap(r[L+1],r[i]);
    }
    // Combine
    for (int i{1}; i < 4; ++i) {
        if (r[i] != 0 && r[i] == r[i-1]) {
            r[i-1] += 1;
            r[i] = 0;
        }
    }
    // Compress again
    for (int i{1}; i < 4; ++i) {
        int L = i-1;
        while (L >= 0 && r[L] == 0) {
            --L;
        }
        std::swap(r[L+1],r[i]);
    }
    return ((r[0] & 0x000F) << 12) | ((r[1] & 0x000F) << 8) | ((r[2] & 0x000F) << 4) | ((r[3] & 0x000F) << 0);
}

constexpr uint32_t calculateScore(uint16_t row) {
    // nasty bitmask + power of 2 
    return (((row&0x000F) == 0) ? 0 : ((1 << ((row & 0x000F) >> 0) )))
         + (((row&0x00F0) == 0) ? 0 : ((1 << ((row & 0x00F0) >> 4) )))
         + (((row&0x0F00) == 0) ? 0 : ((1 << ((row & 0x0F00) >> 8) ))) 
         + (((row&0xF000) == 0) ? 0 : ((1 << ((row & 0xF000) >> 12))));
} 

constexpr std::array<RowEntry, MOVE_COUNT> generateLookupTable() {
    // Too many loop iterations to be done at compile time
    std::array<RowEntry, MOVE_COUNT> lookup_table;
    for (uint32_t i{0}; i < MOVE_COUNT; ++i) {
        lookup_table[i].result = rowShiftLeft(i);
        lookup_table[i].score = calculateScore(lookup_table[i].result);
    }
    return lookup_table;
};
