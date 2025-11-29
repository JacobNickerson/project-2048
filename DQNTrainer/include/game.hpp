#pragma once

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "look_up_table.hpp"
#include "simulator.hpp"

class GameManager {
    public:
        GameManager(bool user_input, uint32_t rng, const RowEntry* look_up_table);
        ~GameManager();
        
        void printBoard();
        Move pollMove();
        bool applyMove(Move move);

    private:
        bool user_input;
        Simulator sim;
};

void populateSharedMemory(bip::managed_shared_memory& shm);