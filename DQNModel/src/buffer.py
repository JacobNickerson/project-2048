import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    Buffer for sampling previous experiences
    """

    def __init__(self, capacity: int, state_shape: Tuple[int, ...]) -> None:
        self.capacity: int = capacity
        self.state_shape: Tuple[int, ...] = state_shape
        self.states: np.ndarray = np.zeros((capacity, *state_shape), dtype=np.ndarray)
        self.next_states: np.ndarray = np.zeros(
            (capacity, *state_shape), dtype=np.ndarray
        )
        self.actions: np.ndarray = np.zeros((capacity,), dtype=np.uint8)
        self.rewards: np.ndarray = np.zeros((capacity,), dtype=np.float64)
        self.dones: np.ndarray = np.zeros((capacity,), dtype=np.bool)
        self.idx: int = 0
        self.size: int = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> None:
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )
