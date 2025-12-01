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
    parser.add_argument("--network", type=str, required=False, help="Path to the common base of the network weights files")
    parser.add_argument("--q-network", type=str, required=False, help="Path to the q-network file")
    parser.add_argument("--target-network", type=str, required=False, help="Path to the target network file")
    parser.add_argument("--average-runs", type=int, required=False, help="Optionally run many runs and collect averages")
    args = parser.parse_args()
    if not args.random:
        if not args.network and not (args.q_network and args.target_network):
            print("error: --network or --q-network and --target-network required if not running with random input")
            return
        if args.network and (args.q_network or args.target_network):
            print("error: --network not compatible with --q-network or --target-network")
            return
        agent = DQNAgent(STATE_DIM, ACTION_DIM)
        if args.network:
            q_network = args.network + "_policy.weights.h5"
            target_network = args.network + "_target.weights.h5"
        else:
            q_network = args.q_network
            target_network = args.target_network
        print(f"Loading Q-Network: {q_network}")
        print(f"Loading Target-Network: {target_network}")
        agent.load_weights(q_network, target_network)
    else:
        agent = RandomAgent(STATE_DIM, ACTION_DIM)
        print("Playing with random inputs")

    match(args.env_type):
        case "py":
            env_man = PyEnvManager(1)
            env = env_man.envs[0]
            if not args.average_runs:
                play_py_dqn(agent,env)
                print("RESULTS")
                print("END STATE: ")
                env.print_board()
                print(f"SCORE: {env.score}")
            else:
                print(f"Averaging across {args.average_runs}")
                scores = []
                max_tiles = []
                end_states = []
                for i in range(args.average_runs):
                    print(f"Game {i} in progress...")
                    env.reset()
                    play_py_dqn(agent,env)
                    unpacked_board = env.get_board(packed=False)
                    scores.append(env.score)
                    max_tiles.append(1 << unpacked_board[np.argmax(unpacked_board)])
                    end_states.append(unpacked_board.copy())
                avg_score = np.mean(scores)
                highest_score = np.max(scores)
                median_max_tile = np.median(max_tiles)
                highest_max_tile = np.max(max_tiles)
                print(f"RESULTS ACROSS {args.average_runs} RUNS")
                print(f"AVG SCORE  : {avg_score}")
                print(f"HIGH SCORE : {highest_score}")
                print(f"MEDIAN TILE: {median_max_tile}")
                print(f"MAX TILE   : {highest_max_tile}")
                with open("output.txt", "a", encoding="utf-8") as f:
                    f.write("score,max_tile,board_state\n")
                    for i in range(args.average_runs):
                        f.write(f"{scores[i]},{max_tiles[i]},{end_states[i]}\n")
        case "cpp":
            # TODO: Implement averaging, API is different between the two ENVs so its not trivial to just replace the env
            env_man = CPPEnvManager(1)
            play_dqn(agent,env_man)
        case "user_input":
            play_user_dqn()
        case _:
            print(f"Environment type {args.env_type} not recognized, valid options: \"py\" and \"cpp\"")
    

if __name__ == "__main__":
    main()