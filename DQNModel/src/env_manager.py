import numpy as np
from multiprocessing import shared_memory, Queue
from typing import List, Tuple
from PySharedMemoryInterface import SharedMemoryInterface
from buffer import State, Action, Reward, Done

message_dtype = np.dtype([
    ('id', np.uint8),
    ('board', np.uint64),
    ('moves', np.uint8),
    ('reward', np.double)
])

class ParallelEnvManager:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.shm = SharedMemoryInterface()

    def write_actions(self, actions):
        for i, a in enumerate(actions):
            if a > 0:
                self.shm.putResponse(i,a)

    def poll_results(self) -> List[int]:  # TODO: Proper type hint
        """
        Batch reads all available states from the shared memory queue
        """
        return np.frombuffer(self.shm.getMessageBatch(), dtype=message_dtype)

    # TODO: Very silly way of doing resets, should make a better way using the manager class
    def reset_all(self):
        print("Resetting all")
        self.poll_results() # clear queue
        for i in range(self.num_envs):
            self.shm.putResponse(i,0b00010000) # send kys signal to all

    def get_all_states(self):
        """
        Gets the states for all processes and puts them in an ordered array
        Should only be called at the start of a training session because it discards all other information
        """
        states = [None]*self.num_envs
        read = 0
        while read < self.num_envs:
            print(f"read = {read}")
            results = self.poll_results()
            for result in results:
                states[result["id"]] = result 
                read += 1
        return states