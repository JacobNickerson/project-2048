import tensorflow as tf
from tensorflow.keras import layers

def create_q_network(state_dim, action_dim):
    model = tf.keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(state_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(action_dim)
    ])
    return model

