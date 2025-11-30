import numpy as np
from src.sim import Simulator, LookupTable 
from src.PySharedMemoryInterface import SharedMemoryInterface # type: ignore
from numpy.typing import NDArray

message_dtype = np.dtype([
    ('id', np.uint8),
    ('board', np.uint64),
    ('moves', np.uint8),
    ('reward', np.float32)
])

pymessage_dtype = np.dtype([
    ('id', np.uint8),
    ('state', np.ndarray),
    ('prev_state', np.ndarray),
    ('moves', np.int64),
    ('reward', np.float64),
    ('is_terminated', np.bool)
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
            res = self.poll_results()
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


# A smarter man would make these two classes inherit an interface or something
class PyEnvManager:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.look_up_table = LookupTable()
        self.envs = np.empty(num_envs,dtype=object)
        for i in range(self.num_envs):
            self.envs[i] = Simulator(
                i,
                self.look_up_table.move_look_up_table,
                self.look_up_table.score_look_up_table
            )

    def write_actions(self, actions):
        """
        Sends valid actions to the environments
        """
        for env, action in zip(self.envs, actions):
            env.make_move(action)

    def poll_results(self) -> np.ndarray:
        """
        Returns an array of experiences from the environments
        """
        arr = ([env.get_experience() for env in self.envs])
        arr = np.array(arr,dtype=pymessage_dtype)
        return arr

    def reset_all(self) -> None:
        """
        Reset all environments
        """
        for env in self.envs:
            env.reset()