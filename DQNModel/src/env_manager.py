import numpy as np
from src.PySharedMemoryInterface import SharedMemoryInterface # type: ignore

message_dtype = np.dtype([
    ('id', np.uint8),
    ('board', np.uint64),
    ('moves', np.uint8),
    ('reward', np.double),
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
        return np.frombuffer(self.shm.getMessageBatch(), dtype=message_dtype)

    def pop_results(self) -> np.ndarray:
        res = self.shm.getMessage()
        if not res:
            return np.zeros(0,dtype=message_dtype)
        arr = np.zeros(1,dtype=message_dtype)
        arr[0]["id"] = res.id
        arr[0]["board"] = res.board
        arr[0]["moves"] = res.moves
        arr[0]["reward"] = res.reward
        return arr
    
    def get_initial_states(self) -> np.ndarray:
        arr = np.zeros(self.num_envs, dtype=message_dtype)
        read = 0
        while read < self.num_envs:
            res = self.pop_results()
            read += len(res)
            for r in res:
                arr[r["id"]] = r
        return arr

    def reset_all(self):
        """
        Reset all environments by clearing the queue and sending reset signal to all envs
        """
        self.poll_results()
        self.write_actions([0b00010000]*self.num_envs) # send kys signal to all