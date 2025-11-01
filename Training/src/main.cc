#include <boost/interprocess/managed_shared_memory.hpp>
#include <iostream>
#include <bitset>
#include "simulator.hpp"
#include "look_up_table.hpp"

using namespace boost::interprocess;

// static std::array<RowEntry, MOVE_COUNT> MOVE_TABLE = generateLookupTable();

int main() {
	// shared_memory_object::remove("proj2048shm"); // just in case of leaks
	// managed_shared_memory shm(create_only, "proj2048shm", 1 << 20);
	// shared_memory_object::remove("proj2048shm");
	// Game testGame(124986745, MOVE_TABLE);
	uint16_t test = 0x2000;
	std::cout << calculateScore(test) << std::endl;

	return 0;
}
