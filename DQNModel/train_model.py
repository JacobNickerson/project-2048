#!/usr/bin/env python3

from argparse import ArgumentParser
import tensorflow as tf
from src.agent import DQNAgent
from src.env_manager import CPPEnvManager, PyEnvManager
from src.train import train_dqn, train_python_dqn

def main():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU")
    else:
        print("GPU not detected")

    parser = ArgumentParser()
    parser.add_argument("--num-env",type=int,required=True)
    parser.add_argument("--epsilon",type=float,required=False,default=0.1)
    parser.add_argument("--env-type",type=str,required=False,default="py")
    parser.add_argument("--ep-save-interval",type=int,required=False,default=100)
    parser.add_argument("--ep-count",type=int,required=False,default=float('inf'))
    args = parser.parse_args()

    STATE_DIM = 16
    ACTION_DIM = 4

    agent_2048 = DQNAgent(STATE_DIM, ACTION_DIM)
    match(args.env_type):
        case "py":
            env_man = PyEnvManager(args.num_env)
            train_python_dqn(agent_2048, env_man, epsilon=args.epsilon, save_every=args.ep_save_interval,episode_count=args.ep_count)
        case "cpp":
            # training_sim = subprocess.Popen(
            #     ["../../DQNTrainer/build/DQNTrainer", f"{NUM_ENVS}"],
            #     stdout = subprocess.PIPE,
            #     stderr = subprocess.PIPE
            # )
            env_man = CPPEnvManager(args.num_env)
            train_dqn(agent_2048, env_man, epsilon=args.epsilon, save_every=args.ep_save_interval,episode_count=args.ep_count)
        case _:
            print(f"Environment type {args.env_type} not recognized, valid options: \"py\" and \"cpp\"")

    
if __name__ == "__main__":
    main()