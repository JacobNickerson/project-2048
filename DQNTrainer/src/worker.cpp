#include "worker.hpp"
#include "lock_free_queue.hpp"
#include "look_up_table.hpp"
#include "shared_memory_structures.hpp"
#include "simulator.hpp"
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

// Unholy constructor
Worker::Worker(
    uint8_t id,
    unsigned rng_seed,
    ProcessControlFlags* control_flags,
    Message* message_buffer,
    LockFreeQueue<Message>* message_queue,
    ResponseCell* DQN_move_array,
    const RowEntry* move_lookup_table
) : 
    id(id),
    simulator(id,1,move_lookup_table),
    control_flags(control_flags),
    message_buffer(message_buffer),
    message_queue(message_queue),
    DQN_move_array(DQN_move_array) {
        bip::scoped_lock<bip::interprocess_mutex> lock(control_flags->mtx);
        while (!control_flags->manager_ready) {
            control_flags->cond.wait(lock, [control_flags]{ return control_flags->workers_ready; });
        }
    }
        

bool Worker::sendReady() {
    // some type of error checking???
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
        auto msg = simulator.generateMessage();
        // queue will always have at least as many spaces, processes can only take one queue space at a time, thus this should never fail
        message_queue->push(message_buffer,simulator.generateMessage());
        while (DQN_move_array[id].read.load()) {
            // wait for DQN to update
            // TODO: Better way to wait than busy waiting
        }
        Move curr_move = DQN_move_array[id].move;
        std::cout << "Got move: " << std::bitset<8>(curr_move) << std::endl;
        DQN_move_array[id].read.store(true);
        simulator.makeMove(curr_move);
        // TODO: Implement method of notifying DQN model that a game has ended
    }
    std::cout << "somehow ended!\n";
}