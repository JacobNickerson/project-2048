#include "look_up_table.hpp"
#include <cassert>

std::array<uint8_t, 4> unpackRow(uint16_t row) {
    std::array<uint8_t, 4> unpacked_row;
    unpacked_row[3] = (row & 0x000F);
    row >>= 4;
    unpacked_row[2] = (row & 0x000F);
    row >>= 4;
    unpacked_row[1] = (row & 0x000F);
    row >>= 4;
    unpacked_row[0] = (row & 0x000F);
    return unpacked_row;
}
uint16_t packRow(std::array<uint8_t,4> row) {
    uint16_t packed_row = 0;
    for (uint8_t i{0}; i < 4; ++i) {
        packed_row |= (row[i] << 4*(3-i));
    }
    return packed_row;
}

std::array<uint16_t,4> transposeBoard(std::array<uint16_t,4> board) {
    uint16_t r0 = board[0];
    uint16_t r1 = board[1];
    uint16_t r2 = board[2];
    uint16_t r3 = board[3];
    std::array<uint16_t,4> transposed = {0,0,0,0};
    
    transposed[0] = ((r0 & 0xF000) >> 0 ) | ((r1 & 0xF000) >> 4) | ((r2 & 0xF000) >> 8) | ((r3 & 0xF000) >> 12);
    transposed[1] = ((r0 & 0x0F00) << 4 ) | ((r1 & 0x0F00) >> 0) | ((r2 & 0x0F00) >> 4) | ((r3 & 0x0F00) >> 8 );
    transposed[2] = ((r0 & 0x00F0) << 8 ) | ((r1 & 0x00F0) << 4) | ((r2 & 0x00F0) >> 0) | ((r3 & 0x00F0) >> 4);
    transposed[3] = ((r0 & 0x000F) << 12) | ((r1 & 0x000F) << 8) | ((r2 & 0x000F) << 4) | ((r3 & 0x000F) >> 0);
    return transposed;
}
uint16_t reverseRow(uint16_t row) {
    uint16_t r = ((row & 0x00FF) << 8) | ((row & 0xFF00) >> 8);
    return ((r & 0x0F0F) << 4) | ((r & 0xF0F0) >> 4);
}

std::pair<uint16_t,int> rowShiftLeft(uint16_t row) {
    // this implementation is disgusting
    int score = 0; 
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
            score += tile_score_lookup[r[i-1]];
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
    return std::make_pair(
        ((r[0] & 0x000F) << 12) | ((r[1] & 0x000F) << 8) | ((r[2] & 0x000F) << 4) | ((r[3] & 0x000F) << 0),
        score
    );
}

std::array<RowEntry, MOVE_COUNT> generateLookupTable() {
    // Too many loop iterations to be done at compile time
    std::array<RowEntry, MOVE_COUNT> lookup_table;
    for (uint32_t i{0}; i < MOVE_COUNT; ++i) {
        auto [result, score_delta] = rowShiftLeft(i);
        lookup_table[i].result = result;
        lookup_table[i].score = score_delta;
    }
    return lookup_table;
};
