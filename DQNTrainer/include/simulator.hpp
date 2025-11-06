#pragma once

#include <array>
#include <cstdint>

#include "rng.hpp"
#include "look_up_table.hpp"
#include "shared_memory_structures.hpp"

using Move = uint8_t;

class Simulator {
    public:
        // Create a Simulator object and initializes the board to a valid 2048 starting state
        Simulator(uint8_t id, uint32_t rng_seed, const RowEntry* MOVE_TABLE);
        // Returns a bit-packed char representing available moves, from LSB to MSB -> LEFT, RIGHT, UP, DOWN, NO MOVES AVAILABLE
        Move getValidMoves() const;
        // Accepts a bit-packed char representing a move, it is assumed that the input is valid ie exactly one legal move
        // Updates the current moveset to represent the new board state and returns the new moveset
        Move makeMove(Move move);
        // Returns the score of the current board 
        uint32_t getScore() const;
        // Generates a message for use in the shared memory queue
        Message generateMessage() const;

        // converter functions, not sure if they're needed but might be helpful
        uint64_t convertBoardToPacked() const;
        std::array<uint8_t,16> convertBoardToUnpacked() const;

    private:
        uint8_t id;
        Move current_moves;
        std::array<uint16_t,4> board; // represented as four bit packed rows, each tile is 4 bits representing the log2 value of the tile
        XorShift32 rng;
        const RowEntry* MOVE_TABLE = nullptr;
        uint32_t score{0};
        bool game_ended{false};

        void init();

        // Shifts the entire board in a direction, merging tiles at most once
        void moveRight();
        void moveDown();
        void moveLeft();
        void moveUp();

        // Assumes that there is at least one valid space, since game should terminate before this can be called
        void generateRandomTile();
        
        // Helpers
        inline uint8_t shiftAmt(uint8_t index) const { return 4*(3-index%4); }
        inline void setValue(uint8_t index, uint8_t val) { board[index/4] |= (val << shiftAmt(index)); } 
        bool rowCanMoveLeft(uint16_t row) const;

};
