#include "manager.hpp"

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <iostream>
#include <string>

#include "look_up_table.hpp"
#include "shared_memory_structures.hpp"
#include "simulator.hpp"

SimulationManager::SimulationManager(uint8_t process_count, bool logging) :
	process_count(process_count),
	shm(bip::create_only, SHARED_MEMORY_NAME, 1 << 20),
	logging(logging) {}

SimulationManager::~SimulationManager() {
	bip::shared_memory_object::remove(SHARED_MEMORY_NAME);
}

void SimulationManager::spawnSimulators() {
	children.reserve(process_count);
	for (int i{0}; i < process_count; ++i) {
		// TODO: Make this more robust than a hardcoded path
		std::vector<std::string> process_args = {"./DQNWorker", std::to_string(i)};
		children.emplace_back(bp::child(process_args));
	}
	if (logging) {
		std::cout << "Created " << std::to_string(process_count) << " workers\n";
	}
}

void SimulationManager::killSimulators() {
	for (auto& child : children) {
		::kill(child.id(), SIGTERM); // wow boost doesn't have a signal function
	}
	for (auto& child : children) {
		child.wait();
	}
}

void SimulationManager::restartDeadSimulators() {
	for (int i{0}; i < children.size(); ++i) {
		auto& child = children[i];
		if (!child.running()) {
			// TODO: Make this more robust than a hardcoded path
			std::vector<std::string> process_args = {"./DQNWorker", std::to_string(i)};
			child = bp::child(process_args); 
			if (logging) {
				std::cout << "Restarted a dead worker\n";
			}
		}
	}
}

bool SimulationManager::startSimulation() {
	populateSharedMemory();
	// don't let the kids go crazy til we're done
	bip::scoped_lock<bip::interprocess_mutex> lock(control_flags->mtx);
	spawnSimulators();
	control_flags->manager_ready = true;
	if (logging) {
		std::cout << "Sent ready signal to simulators\n";
	}

	while (!control_flags->workers_ready) {
		std::cout << "Manager waiting for workers to be ready...\n";
		control_flags->cond.wait(lock, [this]{ return control_flags->workers_ready; });
	}
	if (logging) {
		std::cout << "Received OK from simulators, simulators running\n"; 
	}
	return true;
}

void SimulationManager::populateSharedMemory() {
	control_flags  = shm.construct<ProcessControlFlags>(CONTROL_FLAGS_NAME)(process_count);
	message_array  = shm.construct<Message>(MESSAGE_ARRAY_NAME)[process_count]();
	DQN_move_array = shm.construct<Move>(DQN_MOVE_ARRAY_NAME)[process_count]();
	auto look_up_table = generateLookupTable();
	move_lookup_table = shm.construct<RowEntry>(MOVE_LOOKUP_TABLE_NAME)[MOVE_COUNT]();
	memcpy(move_lookup_table, look_up_table.data(), sizeof(look_up_table));
	if (logging) {
		std::cout << "Initialized shared memory structures\n";
	}
}