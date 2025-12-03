import numpy as np
from src.sim import Simulator, LookupTable, Move
from src.PySharedMemoryInterface import SharedMemoryInterface  # type: ignore
from src.buffer import Experience
from numpy.typing import NDArray
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import threading
import time
from os import path
import http.server
import socketserver

message_dtype = np.dtype(
    [
        ("id", np.uint8),
        ("board", np.uint64),
        ("moves", np.uint8),
        ("reward", np.float64),
    ]
)

pymessage_dtype = np.dtype(
    [
        ("id", np.uint16),
        ("state", np.ndarray),
        ("prev_state", np.ndarray),
        ("moves", np.int64),
        ("reward", np.float64),
        ("is_terminated", np.bool),
    ]
)


class CPPEnvManager:
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
                self.shm.putResponse(i, a)
                actions[i] = -a

    def poll_results(self) -> np.ndarray:
        """
        Returns the entire message array
        """
        return np.frombuffer(self.shm.getMessageBatch(), dtype=message_dtype)

    def pop_results(self) -> np.ndarray:
        res = self.shm.getMessage()
        if not res:
            return np.zeros(0, dtype=message_dtype)
        arr = np.zeros(1, dtype=message_dtype)
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
        self.write_actions([0b00010000] * self.num_envs)  # send kys signal to all


# A smarter man would make these two classes inherit an interface or something
class PyEnvManager:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.look_up_table = LookupTable()
        self.envs = np.empty(num_envs, dtype=object)
        for i in range(self.num_envs):
            self.envs[i] = Simulator(
                i,
                self.look_up_table
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
        return np.array(
            [env.get_experience() for env in self.envs], dtype=pymessage_dtype
        )

    def reset_all(self) -> np.ndarray:
        """
        Reset all environments
        """
        for env in self.envs:
            env.reset()
        return self.poll_results()

    def reset(self, idx: int) -> Experience:
        self.envs[idx].reset()
        return self.envs[idx].get_experience()
        # return np.array([self.envs[idx].get_experience()], dtype=pymessage_dtype)

class WebEnvManager:
    """
    Only intended for use with run model, does not include training data like reward
    """
    def __init__(self):
        # starting 2048 fork
        self.port = 57575
        self.server = None
        self.server_thread = None
        self.__start_server(self.port)
        # let the server cook.

        self.driver = webdriver.Chrome()
        self.driver.get(f"http://localhost:{self.port}/")
        # Wait until GameManager is ready
        while not self.driver.execute_script("return !!window.gm;"):
            time.sleep(0.1)
        self.game_element = self.driver.find_element("tag name", "body")
        self.board = np.zeros(16)
        self.valid_moves = 0
        self.is_terminated = False
        self.__update_board()

    def shut_down(self):
        if self.server:
            self.server.shutdown()
            self.server_thread.join()
            self.server = None
            self.server_thread = None
        self.driver.quit()

    def get_board(self):
        return self.board

    def get_valid_moves(self):
        return self.valid_moves

    def write_action(self, move):
        match(move):
            case Move.UP.value:
                move = Keys.ARROW_UP
            case Move.DOWN.value:
                move = Keys.ARROW_DOWN
            case Move.LEFT.value:
                move = Keys.ARROW_LEFT
            case Move.RIGHT.value:
                move = Keys.ARROW_RIGHT
            case _:
                raise ValueError("Invalid move")
        print("sending keys")
        self.game_element.send_keys(move)
        time.sleep(0.1)  # small delay so browser can update
        self.__update_board()

    def __update_board(self):
        # get board
        js = """
        return window.getBoard()
        """
        self.board = np.array(self.driver.execute_script(js))
        
        # update valid moves
        self.valid_moves = self.__get_valid_moves(self.board)
        self.is_terminated = self.valid_moves == 0

    def __get_valid_moves(self, board: np.ndarray) -> int:
        temp_board = np.array(board).reshape(4,4)
        valid_moves = 0

        def can_move_left(b):
            for row in b:
                for i in range(1,4):
                    if row[i] != 0 and (row[i-1] == 0 or row[i-1] == row[i]):
                        return True
            return False

        def can_move_right(b):
            return can_move_left(np.fliplr(b))

        def can_move_up(b):
            return can_move_left(b.T)

        def can_move_down(b):
            return can_move_right(b.T)

        if can_move_left(temp_board):
            valid_moves |= Move.LEFT.value
        if can_move_right(temp_board):
            valid_moves |= Move.RIGHT.value
        if can_move_up(temp_board):
            valid_moves |= Move.UP.value
        if can_move_down(temp_board):
            valid_moves |= Move.DOWN.value

        return valid_moves
        
    def __start_server(self, port):
        handler_class = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
            *args, directory="../2048-web", **kwargs
        )

        class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        self.server = ThreadedHTTPServer(("", port), handler_class)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print("Starting web server")
        time.sleep(1)
        print(f"Serving at http://localhost:{port}")