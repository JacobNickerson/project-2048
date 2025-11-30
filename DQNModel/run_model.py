#!/usr/bin/env python3

import tensorflow as tf
from argparse import ArgumentParser
from src.agent import DQNAgent
from src.env_manager import ParallelEnvManager
from src.play import play_dqn

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

    parser = ArgumentParser()
    parser.add_argument("--q-network", type=str, required=True, help="Path to the Q-network weights file")
    parser.add_argument("--target-network", type=str, required=True, help="Path to the target network weights file")
    args = parser.parse_args()
    agent_2048 = DQNAgent(STATE_DIM, ACTION_DIM)
    agent_2048.load_weights(args.q_network, args.target_network)
    env_man = ParallelEnvManager(NUM_ENVS)
    play_dqn(agent_2048,env_man)
    

if __name__ == "__main__":
    main()