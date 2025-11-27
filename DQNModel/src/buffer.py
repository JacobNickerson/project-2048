from typing import Tuple
import numpy as np

State = int
Action = int
Reward = float
Done = bool
Experience = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: Tuple[int, ...]) -> None:
        self.capacity: int = capacity
        self.state_shape: Tuple[int, ...] = state_shape
        self.states: np.ndarray = np.zeros((capacity, *state_shape), dtype=np.int64)
        self.actions: np.ndarray = np.zeros((capacity,), dtype=np.int8)
        self.rewards: np.ndarray = np.zeros((capacity,), dtype=np.float64)
        self.next_states: np.ndarray = np.zeros((capacity, *state_shape), dtype=np.int64)
        self.dones: np.ndarray = np.zeros((capacity,), dtype=np.bool)
        self.idx: int = 0
        self.size: int = 0

    def add(self, state: State, action: Action, reward: Reward, next_state: State, done: bool) -> None:
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Experience:
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )
