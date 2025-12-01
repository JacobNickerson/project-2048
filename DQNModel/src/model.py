from typing import List
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

class DQN(Model):
    """
    Generic DQN Model
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_units: List[int] = [256, 256]) -> None:
        super().__init__()
        self.hidden_layers: List[layers.Layer] = [
            layers.Dense(units, activation='elu') for units in hidden_units
        ]
        self.output_layer: layers.Layer = layers.Dense(action_dim, activation=None)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class DuelingDQN(tf.keras.Model):
    """
    More optimized DQN model using dueling DQN
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.hidden = Sequential([
            layers.Dense(512, activation='elu'),
            layers.LayerNormalization(),
            layers.Dense(512, activation='elu'),
        ])

        self.value_stream = Sequential([
            layers.Dense(256, activation='elu'),
            layers.Dense(128, activation='elu'),
            layers.Dense(1)
        ])
        self.adv_stream = Sequential([
            layers.Dense(256, activation='elu'),
            layers.Dense(128, activation='elu'),
            layers.Dense(action_dim)
        ])

    def call(self, x):
        x = self.hidden(x)
        v = self.value_stream(x)
        a = self.adv_stream(x)
        a_mean = tf.reduce_mean(a, axis=1, keepdims=True)
        return v + (a - a_mean)