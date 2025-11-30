#include "game.hpp"

#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <string>
#include <iostream>
#include <bitset>

#include "shared_memory_structures.hpp"

namespace bip = boost::interprocess;

GameManager::GameManager(bool user_input, uint32_t rng, const RowEntry* look_up_table) : user_input(user_input),
sim(0,rng,look_up_table) {}

GameManager::~GameManager() {
    
}


void GameManager::printBoard() {
    auto board = sim.convertCurrentBoardToUnpacked();
    std::cout << "Score: " << std::to_string(sim.getScore()) << '\n'; 
    for (int i{0}; i < 4; ++i) {
        for (int j{0}; j < 4; ++j) {
            printf("[%03d]",board[4*i+j]);
        }
        std::cout << '\n';
    }
}

Move GameManager::pollMove() {
    if (user_input) {
        Move valid_moves = sim.getValidMoves();
        Move move;
        int user_input;
        std::cout << "Valid moves: " << std::bitset<5>(valid_moves) << '\n';
        std::cout << "Directions : NDURL" << '\n';
        while (true) {
            std::cin >> user_input;
            move = 1 << user_input;
            if ((move & valid_moves) != 0) {
                break;
            } else {
                std::cout << '\r';
            }
        }
        return move;
    } else {
        // while (shm_structures.mva[0].read.load()) {
        //     // wait for DQN to update
        //     // TODO: Better way to wait than busy waiting
        // }
        // Move curr_move = shm_structures.mva[0].move;
        // shm_structures.mva[0].read.store(true);
        // return curr_move;
        return 0;
    }
}

bool GameManager::applyMove(Move move) {
    return sim.makeMove(move) == 0b00010000;
}

void populateSharedMemory(bip::managed_shared_memory& shm) {
	auto message_buffer = shm.construct<Message>(MESSAGE_BUFFER_NAME)[std::bit_ceil(1u)]();
	auto message_queue = shm.construct<LockQueue<Message>>(MESSAGE_QUEUE_NAME)(std::bit_ceil(1u));
	auto DQN_move_array = shm.construct<ResponseCell>(DQN_MOVE_ARRAY_NAME)[1]();
	auto look_up_table = generateLookupTable();
	auto move_lookup_table = shm.construct<RowEntry>(MOVE_LOOKUP_TABLE_NAME)[MOVE_COUNT]();
	memcpy(move_lookup_table, look_up_table.data(), sizeof(look_up_table));
}