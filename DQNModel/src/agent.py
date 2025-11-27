import numpy as np
import tensorflow as tf
from .model import create_q_network


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma
        self.q = create_q_network(state_dim, action_dim)
        self.target_q = create_q_network(state_dim, action_dim)
        self.target_q.set_weights(self.q.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def act(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.q(state[np.newaxis], training=False)
        return int(np.argmax(q_values[0]))

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        next_q = self.target_q(next_states)
        max_next_q = tf.reduce_max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        with tf.GradientTape() as tape:
            q_vals = self.q(states)
            one_hot = tf.one_hot(actions, self.action_dim)
            chosen_q = tf.reduce_sum(q_vals * one_hot, axis=1)
            loss = tf.reduce_mean(tf.square(targets - chosen_q))

        grads = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q.trainable_variables))
        return float(loss.numpy())

    def update_target(self):
        self.target_q.set_weights(self.q.get_weights())