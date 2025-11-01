#include "manager.hpp"

SimulationManager::SimulationManager() : shm(bip::create_only, "proj2048shm", 1 << 20) {}

SimulationManager::~SimulationManager() {
	bip::shared_memory_object::remove("proj2048shm");
}