#pragma once

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/process/v1.hpp>
#include <cstdint>
#include <vector>

#include "simulator.hpp"

namespace bip = boost::interprocess;
namespace bp = boost::process::v1;

class SimulationManager {
    public:
        SimulationManager(uint8_t process_count, bool logging=false);
        ~SimulationManager();

        // Simulator managing functions
        void spawnSimulators();
        void killSimulators();
        void restartDeadSimulators();
        bool startSimulation();

        // Shared memory managing functions
        void populateSharedMemory();

    private:
        std::vector<bp::child> children;
        bip::managed_shared_memory shm;
        uint8_t process_count;
        bool initialized_shm_structures = false;
        bool workers_spawned = false;
        bool logging;
        ProcessControlFlags* control_flags = nullptr;
        Message* message_array = nullptr;
        Move* DQN_move_array = nullptr;
        RowEntry* move_lookup_table = nullptr;
}; 