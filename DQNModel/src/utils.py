import numpy as np


def unpack_64bit_state(packed_state: int) -> np.ndarray:
    packed_state = int(packed_state)
    unpacked = np.array(
        [(packed_state >> 4 * i) & 0xF for i in range(16)], dtype=np.int8
    )
    return unpacked
