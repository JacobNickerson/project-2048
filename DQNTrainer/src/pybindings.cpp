#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include "shared_memory_structures.hpp"

#include <unistd.h>

namespace py = pybind11;

using Move = uint8_t;


struct PySharedMemoryInterface {
    PySharedMemoryInterface() : shm(bip::open_only, SHARED_MEMORY_NAME) {
        // Find shared memory objects
        control_flags = shm.find<ProcessControlFlags>(CONTROL_FLAGS_NAME).first;
        if (!control_flags) { throw std::runtime_error("Couldn't find process control flags"); }
        message_array = shm.find<Message>(MESSAGE_ARRAY_NAME).first;
        if (!message_array) { throw std::runtime_error("Couldn't find message buffer"); }
        move_array= shm.find<ResponseCell>(DQN_MOVE_ARRAY_NAME).first;
        if (!move_array) { throw std::runtime_error("Couldn't find move array"); }
        process_count = control_flags->process_count;
        control_flags->DQN_connected = true;
        control_flags->cond.notify_all();
    }
    bip::managed_shared_memory shm;
    ProcessControlFlags* control_flags;
    uint8_t process_count;
    Message* message_array; // pass to message queue because of shared memory nonsense
    ResponseCell* move_array;

    Message getMessage(int id) {
        auto msg = message_array[id];
        message_array[id].is_fresh = false;
        return msg;
    }

    // NOTE: Returns a RAW MEMORY BUFFER, must be cast in Python using a NumPy dtype
    py::array_t<char> getMessageBatch() {
        constexpr auto size = sizeof(Message);
        Message msgs[process_count];
        memcpy(msgs,message_array,process_count*size);
        for (uint8_t i{0}; i < process_count; ++i) {
            message_array[i].is_fresh = false;
        }
        return py::array_t<char>(
            process_count*size,
            reinterpret_cast<char*>(msgs)
        );
    }

    // NOTE: Should only be called when the simulation has for sure read the response
    bool putResponse(int id, Move move) {
        // slow terrible way of waiting for env to read and mark as read
        while (!move_array[id].read.load()) {}
        move_array[id].move = move;
        move_array[id].read.store(false);
        return true;
    }
};

PYBIND11_MODULE(PySharedMemoryInterface, m) {
    py::class_<PySharedMemoryInterface>(m, "SharedMemoryInterface")
        .def(py::init<>())
        .def("getMessage", &PySharedMemoryInterface::getMessage)
        .def("getMessageBatch", &PySharedMemoryInterface::getMessageBatch)
        .def("putResponse", &PySharedMemoryInterface::putResponse);

    py::class_<Message>(m,"Message")
        .def(py::init<>())
        .def_readwrite("id", &Message::id)
        .def_readwrite("board", &Message::board)
        .def_readwrite("moves", &Message::moves)
        .def_readwrite("reward", &Message::reward)
        .def_readwrite("fresh", &Message::is_fresh);
};