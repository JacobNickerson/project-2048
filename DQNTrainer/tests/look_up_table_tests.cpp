#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <doctest/doctest.h>
#include "look_up_table.hpp"

// Tests
TEST_CASE("Row packing") {
    CHECK(packRow({1,2,3,4}) == 0b0001001000110100);
    CHECK(packRow({15,15,15,15}) == 0b1111111111111111);
    CHECK(packRow({0,0,0,0}) == 0b0000000000000000);
    CHECK(packRow({8,4,2,1}) == 0b1000010000100001);
};

TEST_CASE("Row unpacking") {
    auto case1 = unpackRow(0b0001001000110100);
    std::array<uint8_t,4> soln1 = {1,2,3,4};
    CHECK(std::equal(case1.begin(),case1.end(),soln1.begin()));

    auto case2 = unpackRow(0b1111111111111111);
    std::array<uint8_t,4> soln2 = {15,15,15,15};
    CHECK(std::equal(case2.begin(),case2.end(),soln2.begin()));

    auto case3 = unpackRow(0b0000000000000000);
    std::array<uint8_t,4> soln3 = {0,0,0,0};
    CHECK(std::equal(case3.begin(),case3.end(),soln3.begin()));

    auto case4 = unpackRow(0b1000010000100001);
    std::array<uint8_t,4> soln4 = {8,4,2,1};
    CHECK(std::equal(case4.begin(),case4.end(),soln4.begin()));
};

TEST_CASE("Board transposition") {
    std::array<uint16_t,4> board = {
        packRow({0 ,1 ,2 ,3 }),
        packRow({4 ,5 ,6 ,7 }),
        packRow({8 ,9 ,10,11}),
        packRow({12,13,14,15})
    };
    std::array<uint16_t,4> soln = {
        packRow({0 ,4 ,8 ,12}),
        packRow({1 ,5 ,9 ,13}),
        packRow({2 ,6 ,10,14}),
        packRow({3 ,7 ,11,15})
    };
    board = transposeBoard(board);
    auto left = reinterpret_cast<uint64_t*>(board.data());
    auto right = reinterpret_cast<uint64_t*>(soln.data());
    CHECK(*left == *right);
}

TEST_CASE("Row reversal") {
    CHECK(reverseRow(0b1000010111111010) == 0b1010111101011000);
}

TEST_CASE("Slide left direct calculation") {
    // Slides
    CHECK(rowShiftLeft(0x0000).first == (0x0000));
    CHECK(rowShiftLeft(0x1000).first == (0x1000));
    CHECK(rowShiftLeft(0x0100).first == (0x1000));
    CHECK(rowShiftLeft(0x0010).first == (0x1000));
    CHECK(rowShiftLeft(0x0001).first == (0x1000));

    CHECK(rowShiftLeft(0x2100).first == (0x2100));
    CHECK(rowShiftLeft(0x2010).first == (0x2100));
    CHECK(rowShiftLeft(0x2001).first == (0x2100));
    CHECK(rowShiftLeft(0x0210).first == (0x2100));
    CHECK(rowShiftLeft(0x0201).first == (0x2100));
    CHECK(rowShiftLeft(0x0021).first == (0x2100));

    CHECK(rowShiftLeft(0x3210).first == (0x3210));
    CHECK(rowShiftLeft(0x3201).first == (0x3210));
    CHECK(rowShiftLeft(0x3021).first == (0x3210));
    CHECK(rowShiftLeft(0x0321).first == (0x3210));

    CHECK(rowShiftLeft(0x1234).first == (0x1234));

    // Merges
    CHECK(rowShiftLeft(0x1100).first == (0x2000));
    CHECK(rowShiftLeft(0x1010).first == (0x2000));
    CHECK(rowShiftLeft(0x1001).first == (0x2000));
    CHECK(rowShiftLeft(0x0110).first == (0x2000));
    CHECK(rowShiftLeft(0x0101).first == (0x2000));
    CHECK(rowShiftLeft(0x0011).first == (0x2000));

    CHECK(rowShiftLeft(0x1122).first == (0x2300));
    CHECK(rowShiftLeft(0x1102).first == (0x2200));
    CHECK(rowShiftLeft(0x1221).first == (0x1310));
}

TEST_CASE("Slide left direct score calculation") {
    // Slides
    CHECK(rowShiftLeft(0x0000).second == (0));
    CHECK(rowShiftLeft(0x1000).second == (0));
    CHECK(rowShiftLeft(0x0100).second == (0));
    CHECK(rowShiftLeft(0x0010).second == (0));
    CHECK(rowShiftLeft(0x0001).second == (0));

    CHECK(rowShiftLeft(0x2100).second == (0));
    CHECK(rowShiftLeft(0x2010).second == (0));
    CHECK(rowShiftLeft(0x2001).second == (0));
    CHECK(rowShiftLeft(0x0210).second == (0));
    CHECK(rowShiftLeft(0x0201).second == (0));
    CHECK(rowShiftLeft(0x0021).second == (0));

    CHECK(rowShiftLeft(0x3210).second == (0));
    CHECK(rowShiftLeft(0x3201).second == (0));
    CHECK(rowShiftLeft(0x3021).second == (0));
    CHECK(rowShiftLeft(0x0321).second == (0));

    CHECK(rowShiftLeft(0x1234).second == (0));

    // Merges
    CHECK(rowShiftLeft(0x1100).second == (4));
    CHECK(rowShiftLeft(0x1010).second == (4));
    CHECK(rowShiftLeft(0x1001).second == (4));
    CHECK(rowShiftLeft(0x0110).second == (4));
    CHECK(rowShiftLeft(0x0101).second == (4));
    CHECK(rowShiftLeft(0x0011).second == (4));

    CHECK(rowShiftLeft(0x1122).second == (12));
    CHECK(rowShiftLeft(0x1102).second == (4));
    CHECK(rowShiftLeft(0x1221).second == (8));
}

TEST_CASE("Slide left table lookup") {
    auto MOVE_TABLE = generateLookupTable();
    // Slides
    CHECK(MOVE_TABLE[0x0000].result == (0));
    CHECK(MOVE_TABLE[0x1000].result == (0));
    CHECK(MOVE_TABLE[0x0100].result == (0));
    CHECK(MOVE_TABLE[0x0010].result == (0));
    CHECK(MOVE_TABLE[0x0001].result == (0));

    CHECK(MOVE_TABLE[0x2100].result == (0));
    CHECK(MOVE_TABLE[0x2010].result == (0));
    CHECK(MOVE_TABLE[0x2001].result == (0));
    CHECK(MOVE_TABLE[0x0210].result == (0));
    CHECK(MOVE_TABLE[0x0201].result == (0));
    CHECK(MOVE_TABLE[0x0021].result == (0));

    CHECK(MOVE_TABLE[0x3210].result == (0));
    CHECK(MOVE_TABLE[0x3201].result == (0));
    CHECK(MOVE_TABLE[0x3021].result == (0));
    CHECK(MOVE_TABLE[0x0321].result == (0));

    CHECK(MOVE_TABLE[0x1234].result == (0));

    // Merges
    CHECK(MOVE_TABLE[0x1100].result == (4));
    CHECK(MOVE_TABLE[0x1010].result == (4));
    CHECK(MOVE_TABLE[0x1001].result == (4));
    CHECK(MOVE_TABLE[0x0110].result == (4));
    CHECK(MOVE_TABLE[0x0101].result == (4));
    CHECK(MOVE_TABLE[0x0011].result == (4));


    CHECK(MOVE_TABLE[0x1122].result == (12));
    CHECK(MOVE_TABLE[0x1102].result == (4));
    CHECK(MOVE_TABLE[0x1221].result == (8));
}
