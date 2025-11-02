import tensorflow as tf
import keras
from keras import layers
from collections import deque
import random

class DQNModel:
    def __init__(self, units=64, input_shape=(16,), action_space=4, trained=False):
        self.units = units # number of neurons at a particular layer (64 is chosen abritraily, can be changed)
        self.input_shape = input_shape # shape of the input (game state), ex: (16,) means a 1-D vector with 16 elements
        self.action_space = action_space # number of neurons representing Q-values for each action, 4 to represent for UP, DOWN, LEFT, RIGHT
        self.model = self.get_model
        # self.game = 2048game() TO DO: create a 2048 game
        # if (trained):
            # TO DO: retrieve stored model file and load weights from it

    def get_model(self):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=self.input_shape))

        # i don't know what the proper shape the conv layers for a cnn should take
        # -it's probably fine to also just use dense layers and try conv layers later on if this doesn't work
        model.add(layers.Conv2D(self.units, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(self.units, (3, 3), activation='relu', padding='same'))
        model.add(layers.Flatten())

        model.add(layers.Dense(self.units, activation='relu'))
        model.add(layers.Dense(self.action_space))
        model.compile(optimizer=tf.optimizers.Adam(), loss='mse')
        return model
    
# agent will use the experience replay buffer to learn from past sampled experiences
# buffers should store experiences in the format: (state0, action, reward, state1)
class ExpReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size) # once the buffer is full, the oldest experience is automatically popped
    
    def store(self, exp):
        self.buffer.append(exp)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)
    
# the model is trained using the epsilon-greedy algorithm
# gamma determines the worth of future rewards (higher, at a max of 1, the more important they are) vs instant rewards
# def train(gameEnv, policy_network, target_network, replay_buffer, sample_size=64, gamma=0.99, epsilon=1, epsilon_decay=0.01, learning_rate=0.5, episodes=1000):
#     for episode in range(episodes):
#         # board = self.game.new_game() TO DO: new game function for game
#         game_over = False
#         score = 0
#         max_tile = -1
#         reward = -1

#         while not game_over:
            # store a copy of the board now
            # state = deepcopy(board)