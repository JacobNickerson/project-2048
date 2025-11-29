#!/usr/bin/env python3

import tensorflow as tf
from src.agent import DQNAgent
from src.env_manager import ParallelEnvManager
from src.train import train_dqn

def main():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU")
    else:
        print("GPU not detected")

    NUM_ENVS = 1
    STATE_DIM = 16
    ACTION_DIM = 4

    # training_sim = subprocess.Popen(
    #     ["../../DQNTrainer/build/DQNTrainer", f"{NUM_ENVS}"],
    #     stdout = subprocess.PIPE,
    #     stderr = subprocess.PIPE
    # )
    agent_2048 = DQNAgent(STATE_DIM, ACTION_DIM)
    env_man = ParallelEnvManager(NUM_ENVS)

    # tf.profiler.experimental.start("logs/profile")
    train_dqn(agent_2048, env_man, epsilon=0.1)
    # tf.profiler.experimental.stop()
    # training_sim.terminate()
    

if __name__ == "__main__":
    main()