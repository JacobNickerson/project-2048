#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from src.agent import DQNAgent, RandomAgent
from src.env_manager import CPPEnvManager, PyEnvManager
from src.play import play_dqn, play_py_dqn

def main():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU")
    else:
        print("GPU not detected")

    STATE_DIM = 16
    ACTION_DIM = 4

    parser = ArgumentParser()
    parser.add_argument("--random", type=bool, required=False, default=False)
    parser.add_argument("--env-type",type=str,required=False,default="py")
    parser.add_argument("--q-network", type=str, required=False, help="Path to the Q-network weights file")
    parser.add_argument("--target-network", type=str, required=False, help="Path to the target network weights file")
    parser.add_argument("--average-runs", type=int, required=False, help="Optionally run many runs and collect averages")
    args = parser.parse_args()
    if not args.random:
        if not args.q_network or not args.target_network:
            print("error: --q-network and --target-network required if not running with random input")
            return
        agent = DQNAgent(STATE_DIM, ACTION_DIM)
        agent.load_weights(args.q_network, args.target_network)
        print(f"Loaded Q-Network: {args.q_network}")
        print(f"Loaded Target-Network: {args.target_network}")
    else:
        agent = RandomAgent(STATE_DIM, ACTION_DIM)
        print("Playing with random inputs")

    match(args.env_type):
        case "py":
            env_man = PyEnvManager(1)
            env = env_man.envs[0]
            if not args.average_runs:
                play_py_dqn(agent,env)
            else:
                print(f"Averaging across {args.average_runs}")
                scores = []
                max_tiles = []
                for _ in range(args.average_runs):
                    env.reset()
                    play_py_dqn(agent,env)
                    unpacked_board = env.get_board(packed=False)
                    scores.append(env.score)
                    max_tiles.append(1 << unpacked_board[np.argmax(unpacked_board)])
                avg_score = np.mean(scores)
                highest_score = np.max(scores)
                median_max_tile = np.median(max_tiles)
                highest_max_tile = np.max(max_tiles)
                print(f"RESULTS ACROSS {args.average_runs} RUNS")
                print(f"AVG SCORE  : {avg_score}")
                print(f"HIGH SCORE : {highest_score}")
                print(f"MEDIAN TILE: {median_max_tile}")
                print(f"MAX TILE   : {highest_max_tile}")
        case "cpp":
            # TODO: Implement averaging, API is different between the two ENVs so its not trivial to just replace the env
            env_man = CPPEnvManager(1)
            play_dqn(agent,env_man)
        case _:
            print(f"Environment type {args.env_type} not recognized, valid options: \"py\" and \"cpp\"")
    

if __name__ == "__main__":
    main()