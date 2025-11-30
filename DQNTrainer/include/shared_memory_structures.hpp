#pragma once

#include "look_up_table.hpp"
#include "lock_queue.hpp"
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/process/v1.hpp>
#include <cstdint>
#include <iostream>
#include <bitset>

namespace bip = boost::interprocess;

constexpr char SHARED_MEMORY_NAME[] = "proj2048shm";
constexpr char CONTROL_FLAGS_NAME[] = "control_flags";
constexpr char MESSAGE_BUFFER_NAME[] = "DQN_message_buffer";
constexpr char MESSAGE_QUEUE_NAME[] = "DQN_message_queue";
constexpr char DQN_MOVE_ARRAY_NAME[] = "DQN_move_array";
constexpr char MOVE_LOOKUP_TABLE_NAME[] = "move_lookup_table";


struct ProcessControlFlags {
   ProcessControlFlags(uint8_t process_count) : process_count(process_count) {}
   bip::interprocess_mutex mtx;
   bip::interprocess_condition cond; 

   uint8_t process_count;

   bool manager_ready = false;
   bool workers_ready = false;
   bool moves_ready = false;
   bool DQN_connected = false;
   
   std::atomic<uint8_t> workers_waiting = process_count;
   std::atomic<uint8_t> remaining_moves = process_count;

   int test = 10;
};

// Compiler SPECIFIC pragma to enforce no byte padding, ensures packed data
// should be supported on major compilers though including MSVC, GCC, Clang
#pragma pack(push, 1)
struct Message { 
    uint8_t id;
    uint64_t board;
    uint8_t moves;
    double reward;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct ResponseCell { 
    std::atomic<uint32_t> ready = 0; // 0 == not ready, 1 == ready
    uint8_t move;
};
#pragma pack(pop)

void write_slot(ResponseCell *s, uint8_t move);
uint8_t wait_read_slot(ResponseCell *s);

struct SharedMemoryStructures {
    SharedMemoryStructures(bip::managed_shared_memory& shm) {
        pcf = shm.find<ProcessControlFlags>(CONTROL_FLAGS_NAME).first;
        if (!pcf) {
            std::cerr << "Could not find pcf\n";
        }
        mb = shm.find<Message>(MESSAGE_BUFFER_NAME).first;
        if (!mb) {
            std::cerr << "Could not find mb\n";
        }
        mq = shm.find<LockQueue<Message>>(MESSAGE_QUEUE_NAME).first;
        if (!mq) {
            std::cerr << "Could not find mq\n";
        }
        mva = shm.find<ResponseCell>(DQN_MOVE_ARRAY_NAME).first;
        if (!mva) {
            std::cerr << "Could not find mva\n";
        }
        mlut = shm.find<RowEntry>(MOVE_LOOKUP_TABLE_NAME).first;
        if (!mlut) {
            std::cerr << "Could not find mlut\n";
        }
        std::cout << "TEST: " << std::bitset<16>(mlut[1].result) << std::endl;
        std::cout << (size_t)mlut << std::endl;
    }
    ProcessControlFlags* pcf = nullptr;
    Message* mb = nullptr;
    LockQueue<Message>* mq = nullptr;
    ResponseCell* mva = nullptr;
    const RowEntry* mlut = nullptr;
};