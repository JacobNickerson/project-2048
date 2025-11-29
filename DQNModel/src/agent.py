import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from src.model import DQN
from src.buffer import ReplayBuffer
from src.utils import unpack_state

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
            self.q_network: DQN = DQN(state_dim, action_dim)
            self.target_network: DQN = DQN(state_dim, action_dim)
            # NOTE: use a dummy to preload the networks on GPU, since they build lazily
            dummy = tf.zeros((1, state_dim), dtype=tf.float32)
            self.q_network(dummy)
            self.target_network(dummy)

        self.optimizer: optimizers.Optimizer = optimizers.Adam(learning_rate=lr)
        self.replay_buffer: ReplayBuffer = ReplayBuffer(buffer_capacity, (state_dim,))

    def select_action(self, state: int, epsilon: float, valid_actions: int) -> int:
        """
        Select an action using epsilon-greedy, constrained to valid_actions.
        """
        if (valid_actions & 0b00010000) > 0: # no valid moves, game is ended 
            return 0b00010000

        valid_actions_arr = [i for i in range(self.action_dim) if (valid_actions >> i) & 1]

        if np.random.rand() < epsilon:
            # pick randomly among valid actions
            return 1 << np.random.choice(valid_actions_arr)

        state = unpack_state(state)

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
    def update(self) -> None:
        with tf.device("/GPU:0"):
            if self.replay_buffer.size < self.batch_size:
                return

            packed_states, actions, rewards, packed_next_states, dones = self.replay_buffer.sample(self.batch_size)
            states = np.array([unpack_state(s) for s in packed_states], dtype=np.int8)
            next_states = np.array([unpack_state(s) for s in packed_next_states], dtype=np.int8)
            states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
            next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
            dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)

            next_q = tf.reduce_max(self.target_network(next_states_tensor), axis=1)
            target_q = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q

            with tf.GradientTape() as tape:
                q_values = self.q_network(states_tensor)
                action_q = tf.reduce_sum(q_values * tf.one_hot(actions_tensor, self.action_dim), axis=1)
                loss = tf.reduce_mean(tf.square(target_q - action_q))

            grads = tape.gradient(loss, self.q_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def sync_target_network(self) -> None:
        self.target_network.set_weights(self.q_network.get_weights())