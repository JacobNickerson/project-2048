#pragma once

#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdint>
#include "lock_queue.hpp"
#include "look_up_table.hpp"
#include "shared_memory_structures.hpp"
#include "simulator.hpp"

class Worker {
    public:
        Worker(
            uint8_t id,
            unsigned rng_seed,
            ProcessControlFlags* control_flags,
            Message* message_buffer,
            LockQueue<Message>* message_queue,
            ResponseCell* DQN_move_array,
            const RowEntry* move_lookup_table
        );

        // Decrements the shared control flag for unready workers, and sets the ready flag
        // if all workers are ready.
        bool sendReady();

        // Infinite loop until DQN attaches to shared memory and updates flag
        void waitForDQN();

        // Waits for all simulators to be ready, then runs the simulation, will continually output
        // results into message array and accept move inputs from the DQN move array until signalled
        // to stop by the manager
        void simulate();

    private:
        Simulator simulator; 
        uint8_t id;
        ProcessControlFlags* control_flags = nullptr;
        Message* message_buffer = nullptr;
        LockQueue<Message>* message_queue = nullptr;
        ResponseCell* DQN_move_array = nullptr;
};