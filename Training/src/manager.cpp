#include "manager.hpp"

#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <iostream>

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
	if (!initialized_shm_structures) {
		if (logging) {
			std::cout << "error: cannot spawn simulators because shared memory structures are not initialized\n";
		}
		return;
	}
	children.reserve(process_count);
	for (int i{0}; i < process_count; ++i) {
		std::vector<std::string> process_args = {"./build/worker.o", std::to_string(i)};
		children.emplace_back(bp::child(process_args));
	}
	if (logging) {
		std::cout << "Created " << process_count << " workers\n";
	}
	workers_spawned = true;
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
			std::vector<std::string> process_args = {"./build/run_worker.o", std::to_string(i)};
			child = bp::child(process_args); 
			if (logging) {
				std::cout << "Restarted a dead worker\n";
			}
		}
	}
}

bool SimulationManager::startSimulation() {
	if (!workers_spawned) {
		if (logging) {
			std::cout << "Cannot start simulation, workers not spawned\n";
		}
		return false;
	}
	bip::scoped_lock<bip::interprocess_mutex> lock(control_flags->mtx);
	control_flags->manager_ready = true;
	if (logging) {
		std::cout << "Sent a signal to simulators\n";
	}
	if (logging) {
		std::cout << "Waiting for signal from simulators\n";
	}

	while (!control_flags->workers_ready) {
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
	initialized_shm_structures = true;
}