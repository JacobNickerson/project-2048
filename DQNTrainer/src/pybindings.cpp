#include <boost/interprocess/managed_shared_memory.hpp>
#include <cstdint>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include "lock_free_queue.hpp"
#include "shared_memory_structures.hpp"

#include <unistd.h>

namespace py = pybind11;

using Move = uint8_t;


struct PySharedMemoryInterface {
    PySharedMemoryInterface() : shm(bip::open_only, SHARED_MEMORY_NAME) {
        // Find shared memory objects
        control_flags = shm.find<ProcessControlFlags>(CONTROL_FLAGS_NAME).first;
        if (!control_flags) { throw std::runtime_error("Couldn't find process control flags"); }
        message_buffer = shm.find<Message>(MESSAGE_QUEUE_BUFFER_NAME).first;
        if (!message_buffer) { throw std::runtime_error("Couldn't find message buffer"); }
        message_queue = shm.find<LockFreeQueue<Message>>(MESSAGE_QUEUE_NAME).first;
        if (!message_queue) { throw std::runtime_error("Couldn't find message queue"); }
        move_array= shm.find<ResponseCell>(DQN_MOVE_ARRAY_NAME).first;
        if (!move_array) { throw std::runtime_error("Couldn't find move array"); }
        process_count = control_flags->process_count;
    }
    bip::managed_shared_memory shm;
    ProcessControlFlags* control_flags;
    uint8_t process_count;
    Message* message_buffer; // pass to message queue because of shared memory nonsense
    LockFreeQueue<Message>* message_queue;
    ResponseCell* move_array;

    std::optional<Message> getMessage() {
        return message_queue->pop(message_buffer);
    }

    // NOTE: Returns a RAW MEMORY BUFFER, must be cast in Python using a NumPy dtype
    py::array_t<char> getMessageBatch() {
        auto messages = message_queue->popBatch(message_buffer);
        constexpr auto size = sizeof(Message);
        return py::array_t<char>(
            messages.size()*size,
            reinterpret_cast<char*>(messages.data())
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
        .def_readwrite("reward", &Message::reward);
};