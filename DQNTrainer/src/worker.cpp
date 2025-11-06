#include "worker.hpp"
#include "look_up_table.hpp"
#include "simulator.hpp"
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

// Unholy constructor
Worker::Worker(
    uint8_t id,
    unsigned rng_seed,
    ProcessControlFlags* control_flags,
    Message* message_array,
    Move* DQN_move_array,
    const RowEntry* move_lookup_table
) : 
    id(id),
    simulator(id,1,move_lookup_table),
    control_flags(control_flags),
    message_array(message_array),
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
    std::cout << "SENDING IT!\n";
    volatile bool dumb = true;
    while (dumb) {
        continue;
        message_array[id] = simulator.generateMessage();

        if (--control_flags->remaining_messages == 0) { // ASSUME THIS IS RESET BY DQN MODEL
            control_flags->messages_ready = true;
            control_flags->cond.notify_all();
        }
        while (!control_flags->moves_ready) {
            control_flags->cond.wait(lock, [this] { return control_flags->moves_ready; });
        }
        Move curr_move = DQN_move_array[id];
        lock.lock();
        if (--control_flags->remaining_moves == 0) {
            control_flags->moves_ready = false;
            control_flags->remaining_messages = control_flags->process_count;
            control_flags->cond.notify_all();
        }
        lock.unlock();
        simulator.makeMove(curr_move);
        // TODO: Implement method of notifying DQN model that a game has ended
    }
}