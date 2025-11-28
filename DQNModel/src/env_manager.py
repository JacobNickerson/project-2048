import numpy as np
from multiprocessing import shared_memory, Queue
from typing import List, Tuple
from PySharedMemoryInterface import SharedMemoryInterface
from buffer import State, Action, Reward, Done

message_dtype = np.dtype([
    ('id', np.uint8),
    ('board', np.uint64),
    ('moves', np.uint8),
    ('reward', np.double),
    ('is_fresh', np.bool)
])

class ParallelEnvManager:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.shm = SharedMemoryInterface()

    def write_actions(self, actions):
        """
        Sends valid actions to their respective environments
        Marks each action as sent by setting it to negative
        """
        for i, a in enumerate(actions):
            if a > 0:
                self.shm.putResponse(i,a)
                actions[i] = -a

    def poll_results(self) -> np.ndarray:
        """
        Returns the entire message array
        """
        arr = np.frombuffer(self.shm.getMessageBatch(), dtype=message_dtype)
        return arr

    def reset_all(self):
        """
        Reset all environments by clearing the queue and sending reset signal to all envs
        """
        self.poll_results()
        self.write_actions([0b00010000]*self.num_envs) # send kys signal to all