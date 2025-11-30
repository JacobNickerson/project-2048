from src.agent import DQNAgent
from src.env_manager import ParallelEnvManager

def play_dqn(agent: DQNAgent, env_manager: ParallelEnvManager):
    env_manager.reset_all()  # reset all environments at episode start
    while True:
        state = env_manager.pop_results()
        if state.size == 0:
            continue
        action = [agent.select_action(state["board"],0,state["moves"])]
        env_manager.write_actions(action)
        if action == 0b00010000:  # skip terminal states
            break

    print(state)