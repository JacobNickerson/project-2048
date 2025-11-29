#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <random>

#include "game.hpp"
#include "look_up_table.hpp"
#include "shared_memory_structures.hpp"

int main(int argc, char* argv[]) {
    bool user_input{false};
    for (int i{1}; i < argc; ++i) {
        auto arg = std::string(argv[i]);
        if (arg == "--user-input") {
            user_input = true;
        }
    } 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(1,UINT32_MAX);
    auto lookup_table = generateLookupTable();
    GameManager manager(user_input,dist(gen),lookup_table.data());
    bool game_ended{false};
    while (!game_ended) {
        std::cout << "Loop\n";
        manager.printBoard();
        Move move = manager.pollMove();
        std::cout << "Move: " << std::bitset<5>(move) << std::endl;
        game_ended = manager.applyMove(move);
    }

    std::cout << "GAME OVER\n";
	bip::shared_memory_object::remove(SHARED_MEMORY_NAME);
}