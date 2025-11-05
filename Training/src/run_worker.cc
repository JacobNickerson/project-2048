#include "look_up_table.hpp"
#include "shared_memory_structures.hpp"
#include "worker.hpp"

namespace bip = boost::interprocess;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "error: worker requires an id\n";
        return 1;
    }

    auto shm = bip::managed_shared_memory(bip::open_only, SHARED_MEMORY_NAME);

    uint8_t sim_id = std::stoi(argv[1]);
    auto pcf = shm.find<ProcessControlFlags>(CONTROL_FLAGS_NAME);
    if (!pcf.first) {
        std::cerr << "Process control flags not found in worker " << sim_id << '\n';
        return 1;
    }
    auto ma = shm.find<Message>(MESSAGE_ARRAY_NAME);
    if (!ma.first) {
        std::cerr << "Message array not found in worker " << sim_id << '\n';
        return 1;
    }
    auto mva = shm.find<Move>(DQN_MOVE_ARRAY_NAME);
    if (!mva.first) {
        std::cerr << "DQN move array not found in worker " << sim_id << '\n';
        return 1;
    }
    auto mlut = shm.find<RowEntry>(MOVE_LOOKUP_TABLE_NAME);
    if (!mlut.first) {
        std::cerr << "Move lookup table not found in worker " << sim_id << '\n';
        return 1;
    }
    // TODO: Replace 1 with rng seeding
    Worker worker(
        sim_id,
        1,
        pcf.first,
        ma.first,
        mva.first,
        mlut.first
    );
    if (!worker.sendReady()) {
        std::cerr << "error: worker " << sim_id << " could not send ready\n";
    }
    worker.simulate();

    return 0;
}