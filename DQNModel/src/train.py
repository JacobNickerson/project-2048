import subprocess
from agent import DQNAgent
from env_manager import ParallelEnvManager

def train_dqn(agent: DQNAgent, env_manager: ParallelEnvManager, episodes: int, epsilon: float):
    """
    env_manager: your ParallelEnvManager wrapper that communicates with C++ processes.
    """
    num_envs = env_manager.num_envs
    done_flags = [False] * num_envs

    print("Let's train.")
    for ep in range(episodes):
        print(f"Episode: {ep}")
        active_envs = set(range(num_envs))
        env_manager.reset_all()  # reset all environments at episode start
        # 1. Load states for all active environments
        states = env_manager.get_all_states()

        while active_envs:
            # 2. Select actions for all active envs
            actions = [-1]*num_envs
            for i in range(num_envs):
                if not done_flags[i]:
                    action = agent.select_action(states[i]["board"],epsilon,states[i]["moves"])
                    actions[states[i]["id"]] = action
            print(f"Actions: {action}")

            # 3. Write actions to environments
            env_manager.write_actions(actions)

            # 4. Poll results from C++ environments
            results = env_manager.poll_results()  # returns list of (env_idx, next_state, reward, done)

            # 5. Store experiences in replay buffer
            for result in results:
                env_idx, next_state, valid_moves, reward = result
                action = actions[env_idx]
                if action != 0b00010000:  # skip terminal states
                    agent.replay_buffer.add(states[env_idx]["board"], action, reward, next_state, valid_moves == 0b00010000)
                    states[env_idx] = result 
                else:
                    done_flags[env_idx] = True
                    active_envs.discard(env_idx)

            # 6. Update agent
            agent.update()

        # 7. Sync target network at episode end
        agent.sync_target_network()
    print("FINISHED!")


if __name__ == "__main__":
    NUM_ENVS = 2
    STATE_DIM = 1
    ACTION_DIM = 4

    # training_sim = subprocess.Popen(
    #     ["../../DQNTrainer/build/DQNTrainer", f"{NUM_ENVS}"],
    #     stdout = subprocess.PIPE,
    #     stderr = subprocess.PIPE
    # )
    stink_agent = DQNAgent(STATE_DIM, ACTION_DIM)
    stink_env_manager = ParallelEnvManager(NUM_ENVS)

    train_dqn(stink_agent, stink_env_manager, episodes=1, epsilon=0.1)
    # training_sim.terminate()