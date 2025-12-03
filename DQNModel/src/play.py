from time import sleep
from src.agent import DQNAgent
from src.env_manager import CPPEnvManager, PyEnvManager, WebEnvManager
from src.sim import Simulator, LookupTable, Move


def play_dqn(agent: DQNAgent, env_manager: CPPEnvManager):
    env_manager.reset_all()  # reset all environments at episode start
    while True:
        state = env_manager.pop_results()
        if state.size == 0:
            continue
        action = [agent.select_action(state["board"], 0, state["moves"])]
        env_manager.write_actions(action)
        if action == 0b00010000:
            break

    print(state)


def play_py_dqn(agent: DQNAgent, env: PyEnvManager):
    action_count = 0
    while True:
        action = agent.select_action(
            env.get_board(packed=False), 0, env.get_valid_moves()
        )
        env.make_move(action)
        action_count += 1
        if env.is_terminated:
            break


def play_user_dqn():
    look_up_table = LookupTable()
    sim = Simulator(
        0, look_up_table 
    )
    while True:
        sim.print_board()
        print(f"Score: {sim.get_score()}")
        if sim.is_terminated:
            break
        valid_moves = sim.get_valid_moves()
        # lmao this is so bad
        print(
            f"Valid moves: {'W' if valid_moves & Move.UP.value else 'X'}{'A' if valid_moves & Move.LEFT.value else 'X'}{'S' if valid_moves & Move.DOWN.value else 'X'}{'D' if valid_moves & Move.RIGHT.value else 'X'}"
        )
        move = input("Enter move  (WASD): ").lower()
        match (move):
            case "w":
                move = Move.UP
            case "a":
                move = Move.LEFT
            case "s":
                move = Move.DOWN
            case "d":
                move = Move.RIGHT
            case _:
                print("Invalid move")
                continue
        if (move.value & valid_moves) == 0:
            print("Invalid move")
            continue
        sim.make_move(move.value)
    print("Game over!")
    print(f"Score: {sim.get_score()}")
    sim.print_board()

def play_web_dqn(agent: DQNAgent, env: WebEnvManager):
    action_count = 0
    while True:
        try:
            board = env.get_board()
            moves = env.get_valid_moves()
            print(board.reshape(4,4))
            print(format(moves,"04b"))
            action = agent.select_action(
                env.get_board(), 0, env.get_valid_moves()
            )
            print(f"Sending {format(action, "05b")}")
            env.write_action(action)
            action_count += 1
            if env.is_terminated:
                break
        except KeyboardInterrupt:
            break
    print("Game over!")
    print("Shutting down server")
    sleep(3)
    env.shut_down()