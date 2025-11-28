#pragma once

#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/process/v1.hpp>
#include <cstdint>

namespace bip = boost::interprocess;

constexpr char SHARED_MEMORY_NAME[] = "proj2048shm";
constexpr char CONTROL_FLAGS_NAME[] = "control_flags";
constexpr char MESSAGE_ARRAY_NAME[] = "DQN_message_array";
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
    bool is_fresh = true;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct ResponseCell { 
    std::atomic<bool> read = true;
    uint8_t move;
};
#pragma pack(pop)