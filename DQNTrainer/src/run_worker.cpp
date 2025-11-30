#include <random>
#include "lock_queue.hpp"
#include "shared_memory_structures.hpp"
#include "worker.hpp"

namespace bip = boost::interprocess;

bool logging = false;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "error: worker requires an id\n";
        return 1;
    }
    for (int i{2}; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--verbose") {
            logging = true;
        }
    }
    auto shm = bip::managed_shared_memory(bip::open_only, SHARED_MEMORY_NAME);

    uint8_t sim_id = std::stoi(argv[1]);
    
    auto shm_structures = SharedMemoryStructures(shm);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(1,UINT32_MAX);
    auto rng = dist(gen);
    if (logging) {
        std::cout << "Generated random number: " << rng << std::endl;
    }
    Worker worker(
        sim_id,
        rng,
        shm_structures
    );
    if (!worker.sendReady()) {
        std::cerr << "error: worker " << sim_id << " could not send ready\n";
    }
    worker.waitForDQN();
    worker.simulate();

    return 0;
}