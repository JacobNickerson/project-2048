from src.agent import DQNAgent
from src.env_manager import CPPEnvManager, PyEnvManager

def play_dqn(agent: DQNAgent, env_manager: CPPEnvManager):
    env_manager.reset_all()  # reset all environments at episode start
    while True:
        state = env_manager.pop_results()
        if state.size == 0:
            continue
        action = [agent.select_action(state["board"],0,state["moves"])]
        env_manager.write_actions(action)
        if action == 0b00010000:
            break

    print(state)

def play_py_dqn(agent: DQNAgent, env: PyEnvManager):
    action_count = 0
    while True:
        action = agent.select_action(env.get_board(packed=False),0,env.get_valid_moves())
        env.make_move(action)
        action_count += 1
        if env.is_terminated:
            break