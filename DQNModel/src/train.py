import numpy as np
from src.agent import DQNAgent
from src.env_manager import CPPEnvManager, PyEnvManager
from src.utils import unpack_64bit_state
from time import time

def train_dqn(agent: DQNAgent, env_manager: CPPEnvManager, epsilon: float, save_every=1000, episode_count=float('inf'), file_name="model"):
    num_envs = env_manager.num_envs

    episode = 0
    start = time()
    step_count = 0
    env_manager.reset_all()  # reset all environments at episode start
    # Load initial states and set up some book-keeping structures
    states = env_manager.get_initial_states()
    needs_action = [True] * num_envs # used to prevent reprocessing states
    actions = [-1]*num_envs
    while episode < episode_count:
        step_count += num_envs
        for i in np.arange(num_envs):
            if needs_action[i]:
                state = states[i]
                action = agent.select_action(unpack_64bit_state(state["board"]),epsilon,state["moves"])
                actions[states[i]["id"]] = action
                needs_action[i] = False # mark that state as processed

        env_manager.write_actions(actions)

        results = env_manager.poll_results()  # returns list of (env_idx, next_state, reward, done, is_fresh)

        for result in results:
            env_idx, next_state, valid_moves, reward = result
            action = -actions[env_idx] # sent actions will be negated by write_actions
            if action != 0b00010000:  # skip terminal states
                agent.replay_buffer.add(unpack_64bit_state(states[env_idx]["board"]), action, reward, unpack_64bit_state(next_state), valid_moves == 0b00010000)
            else:
                episode += 1
                print(f"episode: {episode}")
            states[env_idx] = result
            needs_action[env_idx] = True # mark this env as needing an action

        agent.update()

        if step_count > 10000:
            step_count = 0
            agent.sync_target_network()

        if episode % save_every == 0:
            q_net_filename = f"saved_models/{file_name}_policy_{episode}.weights.h5"
            target_net_filename = f"saved_models/{file_name}_{episode}.weights.h5"
            agent.q_network.save_weights(q_net_filename)
            agent.target_network.save_weights(target_net_filename)
    end = time()
    print(f"{end-start}s to run {episode_count} episodes")

def train_python_dqn(agent: DQNAgent, env_manager: PyEnvManager, epsilon: float, save_every=1000, episode_count=float('inf'), file_name="model"):
    episode = 0
    env_manager.reset_all()
    start = time()
    results = env_manager.poll_results()
    states, valid_moves = results["state"], results["moves"]
    actions = np.zeros(env_manager.num_envs, dtype=np.uint8)
    step_count = 0
    while episode < episode_count:
        step_count += env_manager.num_envs
        for env in env_manager.envs:
            actions[env.idx] = agent.select_action(states[env.idx],epsilon,valid_moves[env.idx])

        env_manager.write_actions(actions)
        results = env_manager.poll_results()

        for action, result in zip(actions,results):
            env_idx, prev_state, reward, state, is_terminated, moves = result["id"], result["prev_state"], result["reward"], result["state"], result["is_terminated"], result["moves"]
            if not is_terminated:
                agent.replay_buffer.add(prev_state, action, reward, state, is_terminated)
                states[env_idx] = state
                valid_moves[env_idx] = moves
            else:
                episode += 1
                print(f"episodes: {episode}")
                env = env_manager.envs[env_idx]
                env.reset()
                states[env.idx] = env.get_board(False)
                valid_moves[env.idx] = env.get_valid_moves()

        agent.update()

        if step_count > 10000:
            step_count = 0
            agent.sync_target_network()
            

        if episode % save_every == 0:
            q_net_filename = f"saved_models/{file_name}_policy_{episode}.weights.h5"
            target_net_filename = f"saved_models/{file_name}_target_{episode}.weights.h5"
            agent.q_network.save_weights(q_net_filename)
            agent.target_network.save_weights(target_net_filename)
    end = time()
    print(f"{end-start}s to run {episode_count} episodes")
    for env in env_manager.envs:
        print(f"Env {env.id} score: {env.score}")
        env.print_board()