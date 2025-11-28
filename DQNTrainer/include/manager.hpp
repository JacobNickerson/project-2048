#pragma once

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/process/v1.hpp>
#include <cstdint>
#include <vector>

#include "lock_free_queue.hpp"
#include "shared_memory_structures.hpp"
#include "simulator.hpp"

namespace bip = boost::interprocess;
namespace bp = boost::process::v1;

class SimulationManager {
    public:
        SimulationManager(uint8_t process_count, bool logging=false);
        ~SimulationManager();

        // Simulator managing functions
        void killSimulators();
        void restartDeadSimulators();
        bool startSimulation();


    private:
        std::vector<bp::child> children;
        bip::managed_shared_memory shm;
        uint8_t process_count;
        bool logging;
        ProcessControlFlags* control_flags = nullptr;
        Message* message_array = nullptr;
        ResponseCell* DQN_move_array = nullptr;
        RowEntry* move_lookup_table = nullptr;
        
        // Initializing functions
        void spawnSimulators();
        void populateSharedMemory();
}; 