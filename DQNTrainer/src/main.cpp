#include "manager.hpp"

#include <chrono>
#include <csignal>
#include <thread>

std::atomic<bool> simulation_running = true;
SimulationManager* global_manager;

void signal_handler(int signum) {
    simulation_running = false;
    if (global_manager) {
        global_manager->kill();
    }
}

int main(int argc, char** argv) {
    int process_count = 6;
    if (argc > 1) {
        process_count = std::stoi(argv[1]);
    }
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGQUIT, signal_handler);

	bip::shared_memory_object::remove(SHARED_MEMORY_NAME); // in case of hanging shared memory

    SimulationManager manager(process_count, true);
    global_manager = &manager;
    manager.startSimulation();
    while (simulation_running.load()) { 
        manager.restartDeadSimulators();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    manager.killSimulators();

    return 0;
}