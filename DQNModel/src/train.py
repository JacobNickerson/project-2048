import subprocess
from agent import DQNAgent
from env_manager import ParallelEnvManager

def train_dqn(agent: DQNAgent, env_manager: ParallelEnvManager, episodes: int, epsilon: float):
    """
    env_manager: your ParallelEnvManager wrapper that communicates with C++ processes.
    """
    num_envs = env_manager.num_envs

    print("Let's train.")
    for ep in range(episodes):
        print(f"Episode: {ep}")
        active_envs = set(range(num_envs))
        env_manager.reset_all()  # reset all environments at episode start
        # 1. Load states for all active environments
        states = env_manager.get_all_states()
        needs_action = [True] * num_envs # used to prevent reprocessing states
        actions = [-1]*num_envs

        while active_envs:
            # actions = [-1]*num_envs
            # 2. Select actions for all active envs
            for i in active_envs:
                if needs_action[i]:
                    state = states[i]
                    action = agent.select_action(state["board"],epsilon,state["moves"])
                    actions[states[i]["id"]] = action
                    needs_action[i] = False # mark that state as processed

            # 3. Write actions to environments, write actions inverts the sign of each action to mark it as sent
            env_manager.write_actions(actions)

            # 4. Poll results from C++ environments
            results = env_manager.poll_results()  # returns list of (env_idx, next_state, reward, done)

            # 5. Store experiences in replay buffer
            for result in results:
                env_idx, next_state, valid_moves, reward = result
                action = -actions[env_idx] # sent actions will be negated by write_actions
                actions[env_idx] = -1  
                if action != 0b00010000:  # skip terminal states
                    agent.replay_buffer.add(states[env_idx]["board"], action, reward, next_state, valid_moves == 0b00010000)
                    states[env_idx] = result
                    needs_action[env_idx] = True # mark this env as needing an action
                else:
                    active_envs.discard(env_idx)

            # 6. Update agent
            agent.update()

        # 7. Sync target network at episode end
        agent.sync_target_network()
    print("FINISHED!")


if __name__ == "__main__":
    NUM_ENVS = 6
    STATE_DIM = 1
    ACTION_DIM = 4

    # training_sim = subprocess.Popen(
    #     ["../../DQNTrainer/build/DQNTrainer", f"{NUM_ENVS}"],
    #     stdout = subprocess.PIPE,
    #     stderr = subprocess.PIPE
    # )
    agent_2048 = DQNAgent(STATE_DIM, ACTION_DIM)
    env_man = ParallelEnvManager(NUM_ENVS)

    train_dqn(agent_2048, env_man, episodes=3, epsilon=0.1)
    # print("Saving model")
    agent_2048.q_network.save_weights("saved_models/dqn_policy.weights.h5")
    agent_2048.target_network.save_weights("saved_models/dqn_target.weights.h5")
    # training_sim.terminate()