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
    if (argc >= 3 && argv[3] == "--verbose") {
        logging = true;
    }

    auto shm = bip::managed_shared_memory(bip::open_only, SHARED_MEMORY_NAME);

    uint8_t sim_id = std::stoi(argv[1]);
    auto pcf = shm.find<ProcessControlFlags>(CONTROL_FLAGS_NAME);
    if (!pcf.first) {
        std::cerr << "Process control flags not found in worker " << sim_id << '\n';
        return 1;
    }
    auto mb = shm.find<Message>(MESSAGE_BUFFER_NAME);
    if (!mb.first) {
        std::cerr << "Message buffer not found in worker " << sim_id << '\n';
        return 1;
    }
    auto mq = shm.find<LockQueue<Message>>(MESSAGE_QUEUE_NAME);
    if (!mq.first) {
        std::cerr << "Message buffer not found in worker " << sim_id << '\n';
        return 1;
    }
    auto mva = shm.find<ResponseCell>(DQN_MOVE_ARRAY_NAME);
    if (!mva.first) {
        std::cerr << "DQN move array not found in worker " << sim_id << '\n';
        return 1;
    }
    auto mlut = shm.find<RowEntry>(MOVE_LOOKUP_TABLE_NAME);
    if (!mlut.first) {
        std::cerr << "Move lookup table not found in worker " << sim_id << '\n';
        return 1;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(1,UINT32_MAX);
    if (logging) {
        std::cout << "Generated random number: " << dist(gen) << std::endl;
    }
    Worker worker(
        sim_id,
        dist(gen),
        pcf.first,
        mb.first,
        mq.first,
        mva.first,
        mlut.first
    );
    if (!worker.sendReady()) {
        std::cerr << "error: worker " << sim_id << " could not send ready\n";
    }
    worker.waitForDQN();
    worker.simulate();

    return 0;
}