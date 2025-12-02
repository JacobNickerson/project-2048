import random
from enum import Enum
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from src.buffer import Experience


class Move(Enum):
    NOMOVE = 0b00000000
    LEFT = 0b00000001
    RIGHT = 0b00000010
    UP = 0b00000100
    DOWN = 0b00001000


class LookupTable:
    """
    Generates a look up table for 2048, pre-computing all moves left and their score increases
    """

    def __init__(self):
        move_count = 16**4
        self.moves = np.zeros(move_count, dtype=np.uint16)
        self.scores = np.zeros(move_count, dtype=int)
        self.monotonicity = np.zeros(move_count, dtype=float)
        for i in np.arange(move_count, dtype=np.uint16):
            self.moves[i], self.scores[i], self.monotonicity[i] = (
                self.__shift_row_left(i)
            )

    def __shift_row_left(self, row: np.uint16) -> Tuple[np.uint16, int]:
        r = np.array(
            [(row >> 12) & 0xF, (row >> 8) & 0xF, (row >> 4) & 0xF, (row >> 0) & 0xF],
            dtype=np.uint8,
        )
        diff = np.diff(r)
        monotonicity = 1.0 if (np.all(diff >= 0) or np.all(diff <= 0)) else 0.0
        nz = r[r != 0]
        compact = np.zeros(4, np.uint8)
        compact[: len(nz)] = nz
        r = compact

        score = 0
        merged = np.zeros(4, np.bool_)

        for i in range(1, 4):
            if r[i] != 0 and r[i] == r[i - 1] and not merged[i - 1]:
                r[i - 1] += 1
                r[i] = 0
                merged[i - 1] = True
                score += 1 << int(r[i - 1])

        nz = r[r != 0]
        r2 = np.zeros(4, np.uint8)
        r2[: len(nz)] = nz
        r = r2

        new_row = (
            (r[0].astype(np.uint16) << 12)
            | (r[1].astype(np.uint16) << 8)
            | (r[2].astype(np.uint16) << 4)
            | (r[3].astype(np.uint16) << 0)
        )

        return new_row, int(score), monotonicity


class Simulator:
    def __init__(
        self,
        idx: int,
        look_up_table: LookupTable
    ):
        self.idx = idx
        self.look_up_table = look_up_table
        self.prev_board = np.zeros(4, dtype=np.uint16)
        self.board = np.zeros(4, dtype=np.uint16)
        self.prev_score = 0
        self.score = 0
        self.is_terminated = False
        self.valid_moves = Move.NOMOVE.value
        self.rng = random.Random()
        self.reset()

    def make_move(self, move: Move) -> None:
        """
        move: Move enum representing a move to make in the game
        Applies a shift to the board and updates the current/prev board state, score, and is_terminated
        generates a random tile
        """
        self.prev_board = self.board.copy()
        self.prev_score = self.score
        match (move):
            case Move.LEFT.value:
                self.__move_left()
            case Move.RIGHT.value:
                self.__move_right()
            case Move.UP.value:
                self.__move_up()
            case Move.DOWN.value:
                self.__move_down()
            case Move.NOMOVE.value:
                return
            case _:
                raise ValueError("bruh")
        self.__populate_random_cell(self.board)
        self.valid_moves = self.__get_valid_moves(self.board)
        self.is_terminated = self.valid_moves == Move.NOMOVE.value

    def reset(self) -> None:
        """
        Resets the state of the board to an initial state
        """
        self.board = np.zeros(4, dtype=np.uint16)
        self.prev_board = np.zeros(4, dtype=np.uint16)
        self.score = 0
        self.prev_score = 0
        self.is_terminated = False

        self.__populate_random_cell(self.board)
        self.__populate_random_cell(self.board)
        self.valid_moves = self.__get_valid_moves(self.board)

    def get_experience(self) -> Experience:
        """
        Returns a tuple of form: (ID, Current state, Previous State, Valid Moves, Reward, Terminated)
        """
        return (
            self.idx,
            self.__unpack_board(self.board).astype(np.float32),
            self.__unpack_board(self.prev_board).astype(np.float32),
            self.valid_moves,
            self.__get_reward(self.board, self.prev_board),
            self.is_terminated,
        )

    def get_valid_moves(self) -> int:
        """
        Returns a bitpacking representing valid moves for the current board
        """
        return self.__get_valid_moves(self.board)

    def get_score(self) -> int:
        """
        Returns the current score
        """
        return self.score

    def print_board(self) -> None:
        """
        Prints the current board state to the terminal
        """
        cells = self.__unpack_board(self.board)
        cells = np.array([0 if cell == 0 else 1 << cell for cell in cells])
        print(cells.reshape((4, 4)))

    def get_board(self, packed=False):
        """
        Returns the current board as either an array of packed bits or an unpacked array of integers
        """
        if packed:
            return self.board
        else:
            return self.__unpack_board(self.board)

    def __move_left(self) -> None:
        """
        Sets the previous board to the current board, then shifts all cells in the current
        board left and merging where expected
        """
        self.score += self.look_up_table.scores[self.board].sum()
        self.board[:] = self.look_up_table.moves[self.board]

    def __move_right(self) -> None:
        """
        Sets the previous board to the current board, then shifts all cells in the current
        board right and merging where expected
        """
        self.board = self.__reverse_board(self.board)
        self.score += self.look_up_table.scores[self.board].sum()
        self.board[:] = self.look_up_table.moves[self.board]
        self.board = self.__reverse_board(self.board)

    def __move_up(self) -> None:
        """
        Sets the previous board to the current board, then shifts all cells in the current
        board up and merging where expected
        """
        self.board = self.__transpose_board(self.board)
        self.score += self.look_up_table.scores[self.board].sum()
        self.board[:] = self.look_up_table.moves[self.board]
        self.board = self.__transpose_board(self.board)

    def __move_down(self) -> None:
        """
        Sets the previous board to the current board, then shifts all cells in the current
        board down and merging where expected
        """
        self.board = self.__transpose_board(self.board)
        self.board = self.__reverse_board(self.board)
        self.score += self.look_up_table.scores[self.board].sum()
        self.board[:] = self.look_up_table.moves[self.board]
        self.board = self.__reverse_board(self.board)
        self.board = self.__transpose_board(self.board)

    def __reverse_board(self, board: NDArray[np.uint16]) -> NDArray[np.uint16]:
        """
        board: np.ndarray of shape (4,), dtype=np.uint16
        Each element is a 16-bit row of a 4x4 board, 4 bits per cell.
        Returns np.ndarray of shape (4,) with the reversed board.
        """
        r = ((board & 0x00FF) << 8) | ((board & 0xFF00) >> 8)
        return ((r & 0x0F0F) << 4) | ((r & 0xF0F0) >> 4)

    def __transpose_board(self, board: NDArray[np.uint16]) -> NDArray[np.uint16]:
        """
        board: np.ndarray of shape (4,), dtype=np.uint16
        Each element is a 16-bit row of a 4x4 board, 4 bits per cell.
        Returns np.ndarray of shape (4,) with the transposed board.
        """
        r0, r1, r2, r3 = board

        t0 = (
            ((r0 & 0xF000) >> 0)
            | ((r1 & 0xF000) >> 4)
            | ((r2 & 0xF000) >> 8)
            | ((r3 & 0xF000) >> 12)
        )
        t1 = (
            ((r0 & 0x0F00) << 4)
            | ((r1 & 0x0F00) >> 0)
            | ((r2 & 0x0F00) >> 4)
            | ((r3 & 0x0F00) >> 8)
        )
        t2 = (
            ((r0 & 0x00F0) << 8)
            | ((r1 & 0x00F0) << 4)
            | ((r2 & 0x00F0) >> 0)
            | ((r3 & 0x00F0) >> 4)
        )
        t3 = (
            ((r0 & 0x000F) << 12)
            | ((r1 & 0x000F) << 8)
            | ((r2 & 0x000F) << 4)
            | ((r3 & 0x000F) >> 0)
        )

        return np.array([t0, t1, t2, t3], dtype=np.uint16)

    def __populate_random_cell(self, board: NDArray[np.uint16]) -> None:
        """
        board: np.ndarray of shape (4,), dtype=np.uint16
        Selects an empty cell at random and sets it to 1 90% of the time and 2 10% of the time
        """
        tiles = self.__unpack_board(board)
        empty_mask = tiles == 0
        empty_indices = np.flatnonzero(empty_mask)
        count = empty_indices.size
        if count == 0:
            return

        idx = self.rng.randrange(count)
        flat_pos = empty_indices[idx]

        row = flat_pos // 4
        col = flat_pos % 4

        # NOTE: Varying the rng for tile spawn, randomly, to avoid overfitting to fixed spawn rates
        val = 2 if self.rng.random() < 0.88 + 0.04*self.rng.random() else 4

        shift = (3 - col) * 4
        board[row] |= np.uint16((val.bit_length() - 1) << shift)

    def __unpack_board(self, board: NDArray[np.uint16]) -> NDArray[np.uint32]:
        """
        board: np.ndarray of shape (4,), dtype=np.uint16
        Returns the unpacked form of a bit-packed board as a np.ndarray of shape (16,), dtype=np.uint8
        """
        shifts = np.array([12, 8, 4, 0], dtype=np.uint16)  # bits per tile
        unpacked = (board[:, None] >> shifts[None, :]) & 0xF  # shape (4,4)
        return unpacked.ravel().astype(np.uint32)  # flatten to shape (16,)

    def __pack_board(self, board: NDArray[np.uint16]) -> np.uint64:
        arr64 = board.astype(np.uint64)
        return (arr64[0] << 48) | (arr64[1] << 32) | (arr64[2] << 16) | arr64[3]

    def __get_reward(self, current_board: NDArray[np.uint16], prev_board: NDArray[np.uint16]):
        """
        board: np.ndarray of shape (4,), dtype=np.uint16
        prev_board: np.ndarray of shape (4,), dtype=np.uint16
        Returns a reward value based on the current and previous board state
        Currently values: score delta, empty tiles, largest tile, largest tile is in a corner
        """
        score_delta = self.score - self.prev_score
        # 2s into 4 yields ~2.3, 512s into 1024 yields ~11
        scaled_score = np.log2(score_delta + 1)  # log scaling

        monotonicity = self.__compute_monotonicity(current_board)

        unpacked_curr = self.__unpack_board(current_board)
        empty_delta = np.sum(unpacked_curr == 0) - np.sum(self.__unpack_board(prev_board))

        max_tile = np.max(unpacked_curr)
        corner_indices = [0, 3, 12, 15]  # flattened corners
        # max of 4
        corner_bonus = 1.0 if any(unpacked_curr[i]==max_tile for i in corner_indices) else 0.0

        c_corner = 0.25  # Scaling factor for bonus for having max tile in corner
        c_mono = 0.15   # Scaling factor for monotonicity
        c_empty = 0.10  # Scaling factor for empty tiles
        reward = scaled_score + c_empty*empty_delta + c_mono*monotonicity + c_corner*corner_bonus

        return reward


    def __compute_monotonicity(self, board: NDArray[np.uint16]) -> float:
        return self.look_up_table.monotonicity[board].sum() + self.look_up_table.monotonicity[self.__transpose_board(board)].sum()

    def __get_valid_moves(self, board: np.ndarray) -> int:
        """
        board: shape (4,), dtype uint16
        Returns a bitpacking representing valid moves for a given board
        """
        rev = self.__reverse_board(board)
        trans = self.__transpose_board(board)

        left = self.__row_can_move_left(board).any()
        right = self.__row_can_move_left(rev).any()
        up = self.__row_can_move_left(trans).any()
        down = self.__row_can_move_left(self.__reverse_board(trans)).any()

        moves = (left << 0) | (right << 1) | (up << 2) | (down << 3)
        return moves

    def __row_can_move_left(self, row: NDArray[np.uint16]) -> NDArray[np.bool_]:
        """
        row: uint16 or ndarray of shape (...) containing uint16 bitpacked rows.
        Returns boolean array of the same shape indicating whether each row
        has a valid left move.
        """
        row = row.astype(np.uint16)

        t0 = (row >> 12) & 0xF
        t1 = (row >> 8) & 0xF
        t2 = (row >> 4) & 0xF
        t3 = (row >> 0) & 0xF

        c0 = (t0 == 0) & ((t1 | t2 | t3) != 0)
        c1 = (t1 == 0) & ((t2 | t3) != 0)
        c2 = (t2 == 0) & (t3 != 0)

        m0 = (t0 != 0) & (t0 == t1)
        m1 = (t1 != 0) & (t1 == t2)
        m2 = (t2 != 0) & (t2 == t3)

        return c0 | c1 | c2 | m0 | m1 | m2
