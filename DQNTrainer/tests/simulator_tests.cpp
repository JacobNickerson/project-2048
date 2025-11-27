#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "simulator.hpp"

// Tests
TEST_CASE("rowCanMoveLeft") {
    Simulator test(1,1,nullptr);
    CHECK(test.rowCanMoveLeft(0b0010000100000000) == false);
};