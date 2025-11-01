#pragma once

#include <boost/interprocess/managed_shared_memory.hpp>
// #include <boost/process.hpp>
#include <vector>

namespace bip = boost::interprocess;
// namespace bp = boost::process;

class SimulationManager {
    public:
        SimulationManager();
        ~SimulationManager();

        void spawnSimulator();

    private:
        // std::vector<bp::process> children;
        bip::managed_shared_memory shm;
};