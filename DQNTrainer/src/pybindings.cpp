#include <boost/interprocess/managed_shared_memory.hpp>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "shared_memory_structures.hpp"

namespace py = pybind11;

struct PyProcessControlFlags {
    PyProcessControlFlags() : shm(bip::open_only, SHARED_MEMORY_NAME) {
        data = shm.find<ProcessControlFlags>(CONTROL_FLAGS_NAME).first;
        if (!data) { throw std::runtime_error("Couldn't find process control flags"); }
    }
    bip::managed_shared_memory shm;
    ProcessControlFlags* data;

    void test() {
        std::cout << "Test: " << data->test << std::endl;
    }
};

PYBIND11_MODULE(ProcessControls, m) {
    py::class_<PyProcessControlFlags>(m, "PyProcessControlFlags")
        .def(py::init<>())
        .def("test", &PyProcessControlFlags::test);
};