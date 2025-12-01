import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers # type: ignore
from src.model import DQN, DuelingDQN
from src.buffer import ReplayBuffer
from src.utils import unpack_64bit_state

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 100000,
        batch_size: int = 64
    ) -> None:
        self.action_dim: int = action_dim
        self.gamma: float = gamma
        self.batch_size: int = batch_size

        with tf.device("/GPU:0"):
            self.q_network: DuelingDQN = DuelingDQN(state_dim, action_dim)
            self.target_network: DuelingDQN = DuelingDQN(state_dim, action_dim)
            # NOTE: use a dummy to preload the networks on GPU, since they build lazily
            dummy = tf.zeros((1, state_dim), dtype=tf.float32)
            self.q_network(dummy)
            self.target_network(dummy)

        self.optimizer: optimizers.Optimizer = optimizers.Adam(learning_rate=lr)
        self.replay_buffer: ReplayBuffer = ReplayBuffer(buffer_capacity, (state_dim,))

    def load_weights(self, q_net_path: str, target_net_path: str) -> None:
        with tf.device("/GPU:0"):
            self.q_network.load_weights(q_net_path)
            self.target_network.load_weights(target_net_path)

    def select_action(self, state: np.ndarray, epsilon: float, valid_actions: int) -> int:
        """
        Select an action using epsilon-greedy, constrained to valid_actions.
        """
        if (valid_actions & 0b00010000) > 0: # no valid moves, game is ended 
            return 0b00010000

        valid_actions_arr = [i for i in range(self.action_dim) if (valid_actions >> i) & 1]

        if np.random.rand() < epsilon:
            # pick randomly among valid actions
            return 1 << np.random.choice(valid_actions_arr)

        # Forward pass through the network
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        with tf.device("/GPU:0"):
            q_tensor = self.q_network(state_tensor)[0]  # shape: (action_dim,)
        q_values = q_tensor.numpy()

        # Mask invalid actions
        mask = np.full_like(q_values, -np.inf, dtype=np.float32)
        for action in valid_actions_arr:
            mask[action] = q_values[action]

        return 1 << int(np.argmax(mask))

    @tf.function
    def update(self):
        if self.replay_buffer.size < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = tf.convert_to_tensor(states, tf.float32)
        next_states = tf.convert_to_tensor(next_states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.int32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        dones = tf.convert_to_tensor(dones, tf.float32)

        next_action = tf.argmax(self.q_network(next_states), axis=1)
        next_q_target = tf.gather(
            self.target_network(next_states),
            next_action,
            batch_dims=1
        )

        target = rewards + self.gamma * (1.0 - dones) * next_q_target

        with tf.GradientTape() as tape:
            q = self.q_network(states)
            q_selected = tf.gather(q, actions, batch_dims=1)
            loss = tf.keras.losses.huber(target, q_selected)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))


    def sync_target_network(self) -> None:
        self.target_network.set_weights(self.q_network.get_weights())

class RandomAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = action_dim
        self.state_dim = state_dim

    def select_action(self, state: np.ndarray, epsilon: float, valid_actions: int) -> int:
        if (valid_actions & 0b00010000) > 0: # no valid moves, game is ended 
            return 0b00010000
        valid_actions_arr = [i for i in range(self.action_dim) if (valid_actions >> i) & 1]
        return 1 << np.random.choice(valid_actions_arr)