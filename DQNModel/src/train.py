import numpy as np
from src.agent import DQNAgent
from src.env_manager import ParallelEnvManager, PyEnvManager
from src.utils import unpack_64bit_state
from time import time

def train_dqn(agent: DQNAgent, env_manager: ParallelEnvManager, epsilon: float, save_every=1000, episode_count=float('inf')):
    num_envs = env_manager.num_envs

    episode = 0
    start = time()
    while episode < episode_count:
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
                    states[env_idx] = result
                    needs_action[env_idx] = True # mark this env as needing an action
                else:
                    active_envs.discard(env_idx)

            agent.update()

        agent.sync_target_network()

        if episode % save_every == 0:
            q_net_filename = f"saved_models/dqn_policy_ep_{episode}.weights.h5"
            target_net_filename = f"saved_models/dqn_target_ep_{episode}.weights.h5"
            agent.q_network.save_weights(q_net_filename)
            agent.target_network.save_weights(target_net_filename)
    end = time()
    print(f"{end-start}s to run 5 episodes")

def train_python_dqn(agent: DQNAgent, env_manager: PyEnvManager, epsilon: float, save_every=1000, episode_count=float('inf')):
    episode = 0
    start = time()
    while episode < episode_count:
        episode += 1
        print(f"episode: {episode}")
        env_manager.reset_all()
        active_envs = set(env_manager.envs)
        results = env_manager.poll_results()
        states, valid_moves = results["state"], results["moves"]
        actions = np.zeros(env_manager.num_envs, dtype=np.uint8)
        
        loop_count = 0
        while active_envs:
            loop_count += 1
            for env in active_envs:
                actions[env.id] = agent.select_action(states[env.id],epsilon,valid_moves[env.id])

            env_manager.write_actions(actions)
            results = env_manager.poll_results()

            for action, result in zip(actions,results):
                env_idx, prev_state, reward, state, is_terminated, moves = result["id"], result["prev_state"], result["reward"], result["state"], result["is_terminated"], result["moves"]
                if not is_terminated:  # skip terminal states
                    agent.replay_buffer.add(prev_state, action, reward, state, is_terminated)
                    states[env_idx] = state
                    valid_moves[env_idx] = moves
                else:
                    active_envs.discard(env_manager.envs[env_idx])

            agent.update()

        agent.sync_target_network()

        if episode % save_every == 0:
            q_net_filename = f"saved_models/dqn_policy_ep_{episode}.weights.h5"
            target_net_filename = f"saved_models/dqn_target_ep_{episode}.weights.h5"
            agent.q_network.save_weights(q_net_filename)
            agent.target_network.save_weights(target_net_filename)
    end = time()
    print(f"{end-start}s to run 5 episodes")
    for env in env_manager.envs:
        print(f"Env {env.id} score: {env.score}")
        env.print_board()
        print(f"Estimated score: {estimate_score(env.get_board(False))}")