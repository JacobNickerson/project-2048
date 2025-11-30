#pragma once

#include <cstdint>
#include <array>

struct RowEntry {
    uint16_t result;
    int score;
};

// Each tile can have 16 different values, 4 tiles per row = 65536 rows
constexpr uint32_t MOVE_COUNT = (16*16*16*16); 

static constexpr int tile_score_lookup[17] = {
    0, 2, 4, 8, 16, 32, 64, 128,
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
};

std::array<uint8_t, 4> unpackRow(uint16_t row);
uint16_t packRow(std::array<uint8_t,4> row);

uint16_t reverseRow(uint16_t row);
std::array<uint16_t,4> transposeBoard(std::array<uint16_t,4> board);

// returns {row,score increase}
std::pair<uint16_t,int> rowShiftLeft(uint16_t row);
std::array<RowEntry, MOVE_COUNT> generateLookupTable();