cmake_minimum_required(VERSION 3.15)
project(DQNTrainer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

cmake_policy(SET CMP0167 NEW)
find_package(Boost REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

find_package(doctest)

set(SRC_BINARY1
    src/look_up_table.cpp
    src/manager.cpp
    src/simulator.cpp
    src/worker.cpp
    src/main.cpp
)

set(SRC_BINARY2
    src/look_up_table.cpp
    src/manager.cpp
    src/simulator.cpp
    src/worker.cpp
    src/run_worker.cpp
)

set(LOOK_UP_TABLE_TESTS
    src/look_up_table.cpp
    src/manager.cpp
    src/simulator.cpp
    src/worker.cpp
    tests/look_up_table_tests.cpp
)

set(SIMULATOR_TESTS
    src/look_up_table.cpp
    src/manager.cpp
    src/simulator.cpp
    src/worker.cpp
    tests/simulator_tests.cpp
)

set(SRC_PYTHON_MODULE
    src/pybindings.cpp
)

# Add first executable
add_executable(DQNTrainer ${SRC_BINARY1})
target_include_directories(DQNTrainer PRIVATE include)

# Add second executable
add_executable(DQNWorker ${SRC_BINARY2})
target_include_directories(DQNWorker PRIVATE include)

# Add python module
pybind11_add_module(PySharedMemoryInterface ${SRC_PYTHON_MODULE})
target_include_directories(PySharedMemoryInterface PRIVATE include)

# Add test binaries
add_executable(LookUpTableTests ${LOOK_UP_TABLE_TESTS})
target_include_directories(LookUpTableTests PRIVATE include)

add_executable(SimulatorTests ${SIMULATOR_TESTS})
target_include_directories(SimulatorTests PRIVATE include)