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
        Batch reads all available states from the shared memory queue
        CURRENTLY BROKEN
        Some type of race condition bug in C++ library code causes a message to be dropped occasionally
        Results in deadlock as env waits for response and model waits for message 
        """
        return np.frombuffer(self.shm.getMessageBatch(), dtype=message_dtype)

    #TODO: Very silly way of doing resets, should make a better way using the manager class
    def reset_all(self):
        """
        Reset all environments by clearing the queue and sending reset signal to all envs
        """
        print("Resetting all")
        self.poll_results() # clear queue
        for i in range(self.num_envs):
            self.shm.putResponse(i,0b00010000) # send kys signal to all

    def get_all_states(self):
        """
        Gets the states for all processes and puts them in an ordered array
        Ensures that states for all envs are read and synchronized
        """
        states = [None]*self.num_envs
        read = 0
        while read < self.num_envs:
            results = self.poll_results()
            for result in results:
                states[result["id"]] = result
                read += 1
        return states