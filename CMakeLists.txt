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
    DQNTrainer/src/shared_memory_structures.cpp
    DQNTrainer/src/look_up_table.cpp
    DQNTrainer/src/manager.cpp
    DQNTrainer/src/simulator.cpp
    DQNTrainer/src/worker.cpp
    DQNTrainer/src/main.cpp
)

set(SRC_BINARY2
    DQNTrainer/src/shared_memory_structures.cpp
    DQNTrainer/src/look_up_table.cpp
    DQNTrainer/src/manager.cpp
    DQNTrainer/src/simulator.cpp
    DQNTrainer/src/worker.cpp
    DQNTrainer/src/run_worker.cpp
)

set(SRC_BINARY3
    DQNTrainer/src/shared_memory_structures.cpp
    DQNTrainer/src/look_up_table.cpp
    DQNTrainer/src/simulator.cpp
    DQNTrainer/src/game.cpp
    DQNTrainer/src/run_game.cpp
)



set(SRC_PYTHON_MODULE
    DQNTrainer/src/shared_memory_structures.cpp
    DQNTrainer/src/pybindings.cpp
)

# Add first executable
add_executable(DQNTrainer ${SRC_BINARY1})
target_include_directories(DQNTrainer PRIVATE DQNTrainer/include)

# Add second executable
add_executable(DQNWorker ${SRC_BINARY2})
target_include_directories(DQNWorker PRIVATE DQNTrainer/include)

# Add second executable
add_executable(DQNPlayer ${SRC_BINARY3})
target_include_directories(DQNPlayer PRIVATE DQNTrainer/include)

# Add python module
pybind11_add_module(PySharedMemoryInterface ${SRC_PYTHON_MODULE})
target_include_directories(PySharedMemoryInterface PRIVATE DQNTrainer/include)
set_target_properties(PySharedMemoryInterface PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/DQNModel/src
)

# Add custom command to generate stub
if (GENERATE_PY_STUB)
    add_custom_command(
        TARGET PySharedMemoryInterface POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/DQNModel/src
                python3 -m pybind11_stubgen PySharedMemoryInterface -o ${CMAKE_CURRENT_SOURCE_DIR}/DQNModel/src
        COMMENT "Generating .pyi stubs for mylib"
    )
endif()

if (ENABLE_TESTS)
    set(LOOK_UP_TABLE_TESTS
        DQNTrainer/src/look_up_table.cpp
        DQNTrainer/tests/look_up_table_tests.cpp
    )

    set(SIMULATOR_TESTS
        DQNTrainer/src/look_up_table.cpp
        DQNTrainer/src/simulator.cpp
        DQNTrainer/tests/simulator_tests.cpp
    )
    # Add test binaries
    add_executable(LookUpTableTests ${LOOK_UP_TABLE_TESTS})
    target_include_directories(LookUpTableTests PRIVATE DQNTrainer/include)

    add_executable(SimulatorTests ${SIMULATOR_TESTS})
    target_include_directories(SimulatorTests PRIVATE DQNTrainer/include)
endif()