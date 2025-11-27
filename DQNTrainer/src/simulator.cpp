#include "simulator.hpp"
#include "look_up_table.hpp"
#include <cstdint>
#include <cstring>

constexpr uint8_t LEFT   = 0b00000001;
constexpr uint8_t RIGHT  = 0b00000010;
constexpr uint8_t UP     = 0b00000100;
constexpr uint8_t DOWN   = 0b00001000;
constexpr uint8_t NOMOVE = 0b00010000;

Simulator::Simulator(uint8_t id, uint32_t rng_seed, const RowEntry* MOVE_TABLE) : id(id), rng(rng_seed), MOVE_TABLE(MOVE_TABLE) {
    init();
    return;
}
Move Simulator::makeMove(Move move) {
    prev_board = board;
    switch (move) {
        case LEFT:   { moveLeft();  break; }
        case RIGHT:  { moveRight(); break; }
        case UP:     { moveUp();    break; }
        case DOWN:   { moveDown();  break; }
        case NOMOVE: { init(); return getValidMoves(); } // Sent by model as a signal to restart
        default: { game_ended = true; return NOMOVE; }
    }
    generateRandomTile();
    current_moves = getValidMoves();
    return current_moves;
}
uint32_t Simulator::getScore() const {
    return score;
}

void Simulator::moveRight() {
    score = 0;
    for (auto& row : board) {
        row = reverseRow(MOVE_TABLE[reverseRow(row)].result);
        score += MOVE_TABLE[row].score;
    }
}
void Simulator::moveDown() {
    score = 0;
    board = transposeBoard(board);
    for (auto& row : board) {
        row = reverseRow(MOVE_TABLE[reverseRow(row)].result);
        score += MOVE_TABLE[row].score;
    }
    board = transposeBoard(board);
}
void Simulator::moveLeft() {
    score = 0;
    for (auto& row : board) {
        row = MOVE_TABLE[row].result;
        score += MOVE_TABLE[row].score;
    }
}
void Simulator::moveUp() {
    score = 0;
    board = transposeBoard(board);
    for (auto& row : board) {
        row = MOVE_TABLE[row].result;
        score += MOVE_TABLE[row].score;
    }
    board = transposeBoard(board);
}

void Simulator::generateRandomTile() {
    uint8_t empty[16];
    uint32_t count{0};
    for (uint8_t i{0}; i < 4; ++i) {
        for (uint8_t j{0}; j < 4; ++j) {
            uint8_t cell = ((board[i] >> 4*(3-j)) & (0x000F)) == 0; 
            empty[count] = (i << 4) | (j & 0x0F); // MS four bits row, LS four bits col
            count += cell; // only advance when finding an empty tile
        }
    }
    auto ind = rng.nextUInt(count);
    int rng_roll = rng.nextUInt(10);
    auto val = 1 << (int)(rng_roll == 9);
    uint8_t row = ((empty[ind]>>4) & 0x0F); 
    uint8_t col = (empty[ind]&0x0F);
    board[row] |= (val << ((3-col) * 4)); 
    score += (1 << val);
}

void Simulator::init() {
    game_ended = false;
    score = 0;
    memset(board.data(), 0, sizeof(board));
    generateRandomTile();
    generateRandomTile();
    current_moves = getValidMoves();
    prev_board = board;
}
uint64_t Simulator::convertBoardToPacked() const {
    uint64_t converted = 0;
    for (int i{0}; i < 4; ++i) {
        converted |= ((uint64_t)board[i] << 16*(3-i));
    }
    return converted;
}
std::array<uint8_t,16> Simulator::convertBoardToUnpacked() const {
    std::array<uint8_t,16> converted;
    memset(converted.data(), 0, sizeof(converted));
    for (int i{0}; i < 4; ++i) {
        uint16_t curr = board[i];
        converted[(4*i)+3] = curr & 0x000F;
        curr >>= 4;
        converted[(4*i)+2] = curr & 0x000F;
        curr >>= 4;
        converted[(4*i)+1] = curr & 0x000F;
        curr >>= 4;
        converted[(4*i)+0] = curr & 0x000F;
    }
    return converted;
}
bool Simulator::rowCanMoveLeft(uint16_t row) const {
    uint8_t t0 = (row >> 12) & 0xF;
    uint8_t t1 = (row >> 8) & 0xF;
    uint8_t t2 = (row >> 4) & 0xF;
    uint8_t t3 = (row >> 0) & 0xF;

    return ((t0 == 0 && (t1|t2|t3)) |
            (t1 == 0 && (t2|t3)) |
            (t2 == 0 && t3) |
            (t0 && t0 == t1) |
            (t1 && t1 == t2) |
            (t2 && t2 == t3));
}
Move Simulator::getValidMoves() const {
    Move valid_moves = 0;
    for (const auto& row : board) {
        // left
        valid_moves |= rowCanMoveLeft(row) << 0;
        // right
        valid_moves |= rowCanMoveLeft(reverseRow(row)) << 1;
    }
    auto temp = transposeBoard(board);
    for (const auto& row : temp) {
        // up
        valid_moves |= rowCanMoveLeft(row) << 2;
        // down
        valid_moves |= rowCanMoveLeft(reverseRow(row)) << 3;
    }
    // return moves or no moves flag if no available moves
    return valid_moves + ((valid_moves == 0) * 0b00010000);
} 
Message Simulator::generateMessage() const {
    return { 
        id,
        convertBoardToPacked(),
        current_moves,
        getReward()
    };
}

double Simulator::getReward() const {
    // naive reward: difference in score
    uint32_t prev_score = 0;
    for (const auto& row : prev_board) {
        prev_score += calculateScore(row);
    }
    return static_cast<double>(score-prev_score);
}