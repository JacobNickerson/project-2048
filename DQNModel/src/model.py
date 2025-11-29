from typing import List
import tensorflow as tf
from tensorflow.keras import layers, Model 

class DQN(Model):
    def __init__(self, state_dim: int, action_dim: int, hidden_units: List[int] = [128, 128]) -> None:
        super().__init__()
        self.hidden_layers: List[layers.Layer] = [
            layers.Dense(units, activation='relu') for units in hidden_units
        ]
        self.output_layer: layers.Layer = layers.Dense(action_dim, activation=None)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
