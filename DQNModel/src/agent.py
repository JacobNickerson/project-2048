import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers  # type: ignore

from src.model import DQN, DuelingDQN
from src.buffer import ReplayBuffer
from src.utils import unpack_64bit_state
from src.sim import Move


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
    ) -> None:
        self.action_dim: int = action_dim
        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.loss_fn = tf.keras.losses.Huber(delta=1.0)
        self.decay_steps = 2_000_000

        with tf.device("/GPU:0"):
            self.q_network: DuelingDQN = DuelingDQN(state_dim, action_dim)
            self.target_network: DuelingDQN = DuelingDQN(state_dim, action_dim)
            # NOTE: use a dummy to preload the networks on GPU, since they build lazily
            dummy = tf.zeros((1, state_dim), dtype=tf.float32)
            self.q_network(dummy)
            self.target_network(dummy)

        self.optimizer: optimizers.Optimizer = optimizers.Adam(
            learning_rate=lr, clipnorm=5.0
        )
        self.replay_buffer: ReplayBuffer = ReplayBuffer(buffer_capacity, (state_dim,))

    def load_weights(self, q_net_path: str, target_net_path: str) -> None:
        with tf.device("/GPU:0"):
            self.q_network.load_weights(q_net_path)
            self.target_network.load_weights(target_net_path)

    def select_action(
        self, state: np.ndarray, epsilon: float, valid_actions: int
    ) -> int:
        """
        Epsilon-greedy action selection using valid_actions bitflags.
        """
        if (
            valid_actions == 0
        ):  # no valid moves, game ended, shouldn't ever happen because ended games will restart before calling this
            raise ValueError("HOW")

        valid_actions_arr = np.flatnonzero(
            [(valid_actions >> i) & 1 for i in range(self.action_dim)]
        )

        if np.random.rand() < epsilon:
            return 1 << np.random.choice(valid_actions_arr)

        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        with tf.device("/GPU:0"):
            q_values = self.q_network(state_tensor)[0].numpy()

        mask = np.full_like(q_values, -np.inf)
        mask[valid_actions_arr] = q_values[valid_actions_arr]

        # Return as bitflag
        return 1 << int(np.argmax(mask))

    def select_actions_batch(
        self, states: np.ndarray, epsilon: float, valid_actions_list: np.ndarray
    ) -> np.ndarray:
        """
        Batch epsilon-greedy action selection.
        states: shape (num_envs, state_dim)
        valid_actions_list: array of int bitflags, shape (num_envs,)
        Returns: array of actions as bitflags, shape (num_envs,)
        """
        num_envs = states.shape[0]
        actions = np.zeros(num_envs, dtype=np.uint8)

        # forward pass for all states at once
        state_tensor = tf.convert_to_tensor(
            np.stack(states).astype(np.float32), dtype=tf.float32
        )
        with tf.device("/GPU:0"):
            q_values_batch = self.q_network(
                state_tensor
            ).numpy()  # shape: (num_envs, action_dim)

        for i in range(num_envs):
            valid_actions = valid_actions_list[i]

            if valid_actions == Move.NOMOVE.value:
                actions[i] = Move.NOMOVE.value
                continue

            # convert bitflags to indices
            valid_indices = np.flatnonzero(
                [(valid_actions >> j) & 1 for j in range(self.action_dim)]
            )

            if np.random.rand() < epsilon:
                actions[i] = 1 << np.random.choice(valid_indices)
                continue

            mask = np.full_like(q_values_batch[i], -np.inf)
            mask[valid_indices] = q_values_batch[i, valid_indices]

            actions[i] = 1 << int(np.argmax(mask))

        return actions

    def update(self) -> bool:
        """
        Updates the model gradients
        """
        if self.replay_buffer.size < self.batch_size:
            return False

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        actions_idx = tf.cast(
            tf.math.log(tf.cast(actions, tf.float32)) / tf.math.log(2.0), tf.int32
        )
        self.__update_step(
            tf.convert_to_tensor(states, tf.float32),
            tf.convert_to_tensor(next_states, tf.float32),
            tf.convert_to_tensor(actions_idx, tf.int32),
            tf.convert_to_tensor(rewards, tf.float32),
            tf.convert_to_tensor(dones, tf.float32),
        )
        return True

    @tf.function
    def __update_step(self, states, next_states, actions, rewards, dones) -> None:
        """
        Helper function for gradient updates, separates all tensorflow functions into a single function for GPU utilization
        """
        next_action = tf.cast(tf.argmax(self.q_network(next_states), axis=1), tf.int32)
        next_q_target = tf.gather(
            self.target_network(next_states), next_action, batch_dims=1
        )

        target = rewards + self.gamma * (1.0 - dones) * next_q_target

        with tf.GradientTape() as tape:
            q = self.q_network(states)
            q_selected = tf.gather(q, actions, batch_dims=1)
            loss = self.loss_fn(target, q_selected)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def sync_target_network(self) -> None:
        """
        Syncs the model's target network with its q-network
        """
        self.target_network.set_weights(self.q_network.get_weights())


class RandomAgent:
    """
    Agent that supplies random moves
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim
        self.state_dim = state_dim

    def select_action(
        self, state: np.ndarray, epsilon: float, valid_actions: int
    ) -> int:
        if (valid_actions & 0b00010000) > 0:  # no valid moves, game is ended
            return 0b00010000
        valid_actions_arr = [
            i for i in range(self.action_dim) if (valid_actions >> i) & 1
        ]
        return 1 << np.random.choice(valid_actions_arr)


class UserAgent:
    """
    Agent that gets moves from user input in the terminal
    """

    def __init__(self):
        self.valid_moves = 0

    def select_action(
        self, state: np.ndarray, epsilon: float, valid_actions: int
    ) -> int:
        if (valid_actions & 0b00010000) > 0:  # no valid moves, game is ended
            return 0b00010000
        # lmao this is so bad
        print(
            f"Valid moves: {'W' if valid_actions & Move.UP.value else 'X'}{'A' if valid_actions & Move.LEFT.value else 'X'}{'S' if valid_actions & Move.DOWN.value else 'X'}{'D' if valid_actions & Move.RIGHT.value else 'X'}"
        )
        while True:
            move = input("Enter move  (WASD): ").lower()
            match (move):
                case "w":
                    move = Move.UP
                    break
                case "a":
                    move = Move.LEFT
                    break
                case "s":
                    move = Move.DOWN
                    break
                case "d":
                    move = Move.RIGHT
                    break
                case _:
                    print("Invalid move")
            if (move.value & self.valid_moves) == 0:
                print("Invalid move")
        return move.value
