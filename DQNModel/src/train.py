import subprocess
from agent import DQNAgent
from env_manager import ParallelEnvManager
import tensorflow as tf
from time import time

def train_dqn(agent: DQNAgent, env_manager: ParallelEnvManager, epsilon: float):
    num_envs = env_manager.num_envs

    episode = 0
    while True:
        episode += 1
        print(f"episode: {episode}")
        active_envs = set(range(num_envs))
        env_manager.reset_all()  # reset all environments at episode start
        
        # Load initial states and set up some book-keeping structures
        states = env_manager.get_initial_states()
        needs_action = [True] * num_envs # used to prevent reprocessing states
        actions = [-1]*num_envs

        while active_envs:
            for i in active_envs:
                if needs_action[i]:
                    state = states[i]
                    action = agent.select_action(state["board"],epsilon,state["moves"])
                    actions[states[i]["id"]] = action
                    needs_action[i] = False # mark that state as processed

            env_manager.write_actions(actions)

            results = env_manager.pop_results()  # returns list of (env_idx, next_state, reward, done, is_fresh)

            for result in results:
                env_idx, next_state, valid_moves, reward = result
                action = -actions[env_idx] # sent actions will be negated by write_actions
                if action != 0b00010000:  # skip terminal states
                    agent.replay_buffer.add(states[env_idx]["board"], action, reward, next_state, valid_moves == 0b00010000)
                    states[env_idx] = result
                    needs_action[env_idx] = True # mark this env as needing an action
                else:
                    active_envs.discard(env_idx)

            agent.update()

        agent.sync_target_network()

        if episode % 100 == 0:
            q_net_filename = f"saved_models/dqn_policy_ep_{episode}.weights.h5"
            target_net_filename = f"saved_models/dqn_target_ep_{episode}.weights.h5"
            agent.q_network.save_weights(q_net_filename)
            agent.target_network.save_weights(target_net_filename)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU")
    else:
        print("GPU not detected")

    NUM_ENVS = 1
    NUM_EP = 20
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
