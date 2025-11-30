#include "worker.hpp"
#include "lock_queue.hpp"
#include "shared_memory_structures.hpp"
#include "simulator.hpp"
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

// Unholy constructor
Worker::Worker(
    uint8_t id,
    unsigned rng_seed,
    SharedMemoryStructures shm_structures
) : 
    id(id),
    simulator(id,rng_seed,shm_structures.mlut),
    control_flags(shm_structures.pcf),
    message_buffer(shm_structures.mb),
    message_queue(shm_structures.mq),
    DQN_move_array(shm_structures.mva) {
        bip::scoped_lock<bip::interprocess_mutex> lock(control_flags->mtx);
        while (!control_flags->manager_ready) {
            control_flags->cond.wait(lock, [this]{ return control_flags->workers_ready; });
        }
    }
        
void Worker::waitForDQN() {
    bip::scoped_lock<bip::interprocess_mutex> lock(control_flags->mtx);
    while (!control_flags->DQN_connected) {
        control_flags->cond.wait(lock, [this]{ return control_flags->DQN_connected; });
    }
}

bool Worker::sendReady() {
    // some type of error checking???
    std::cout << "Sending ready!\n";
    bip::scoped_lock<bip::interprocess_mutex> lock(control_flags->mtx);
    if (--control_flags->workers_waiting == 0) {
        control_flags->workers_ready = true;
        control_flags->cond.notify_all();
    } 
    return true;
}

void Worker::simulate() {
    bip::scoped_lock<bip::interprocess_mutex> lock(control_flags->mtx);
    while (!control_flags->workers_ready) {
        control_flags->cond.wait(lock, [this] { return control_flags->workers_ready; });
    }
    lock.unlock();
    for (;;) {
        // queue will always have at least as many spaces, processes can only take one queue space at a time, thus this should never fail
        auto msg = simulator.generateMessage();
        while (!message_queue->push(message_buffer, msg)) {} // spin-lock on attempting to push
        Move curr_move = wait_read_slot(&DQN_move_array[id]);
        simulator.makeMove(curr_move);
    }
    std::cout << "somehow ended!\n";
}