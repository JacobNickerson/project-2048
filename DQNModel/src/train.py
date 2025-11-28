import subprocess
from agent import DQNAgent
from env_manager import ParallelEnvManager
from time import time

def train_dqn(agent: DQNAgent, env_manager: ParallelEnvManager, episodes: int, epsilon: float):
    """
    env_manager: your ParallelEnvManager wrapper that communicates with C++ processes.
    """
    num_envs = env_manager.num_envs

    for ep in range(episodes):
        active_envs = set(range(num_envs))
        env_manager.reset_all()  # reset all environments at episode start
        
        # Load initial states and set up some book-keeping structures
        states = env_manager.poll_results()
        print(states)
        needs_action = [True] * num_envs # used to prevent reprocessing states
        actions = [-1]*num_envs

        while active_envs:
            for i in active_envs:
                state = states[i]
                if needs_action[i] and state["is_fresh"]:
                    action = agent.select_action(state["board"],epsilon,state["moves"])
                    actions[states[i]["id"]] = action
                    needs_action[i] = False # mark that state as processed

            env_manager.write_actions(actions)

            results = env_manager.poll_results()  # returns list of (env_idx, next_state, reward, done, is_fresh)

            for result in results:
                env_idx, next_state, valid_moves, reward, is_fresh = result
                if not is_fresh:
                    continue
                action = -actions[env_idx] # sent actions will be negated by write_actions
                actions[env_idx] = -1
                if action != 0b00010000:  # skip terminal states
                    agent.replay_buffer.add(states[env_idx]["board"], action, reward, next_state, valid_moves == 0b00010000)
                    states[env_idx] = result
                    needs_action[env_idx] = True # mark this env as needing an action
                else:
                    active_envs.discard(env_idx)

            agent.update()

        # 7. Sync target network at episode end
        agent.sync_target_network()
    print("FINISHED!")


if __name__ == "__main__":
    NUM_ENVS = 6
    NUM_EP = 3
    STATE_DIM = 1
    ACTION_DIM = 4

    # training_sim = subprocess.Popen(
    #     ["../../DQNTrainer/build/DQNTrainer", f"{NUM_ENVS}"],
    #     stdout = subprocess.PIPE,
    #     stderr = subprocess.PIPE
    # )
    agent_2048 = DQNAgent(STATE_DIM, ACTION_DIM)
    env_man = ParallelEnvManager(NUM_ENVS)

    print("Timing training")
    start = time()
    train_dqn(agent_2048, env_man, episodes=NUM_EP, epsilon=0.1)
    end = time()
    print(f"Elapsed time for {NUM_ENVS} training for {NUM_EP} episodes: {end-start} seconds")
    agent_2048.q_network.save_weights("saved_models/dqn_policy.weights.h5")
    agent_2048.target_network.save_weights("saved_models/dqn_target.weights.h5")
    # training_sim.terminate()