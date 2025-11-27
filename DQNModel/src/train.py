import numpy as np
from PySharedMemoryInterface import SharedMemoryInterface
from .buffer import ReplayBuffer
from .agent import DQNAgent


def train():
    env = SharedMemoryInterface()
    print("Hehe haha")
    return
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    buffer = ReplayBuffer()

    episodes = 200
    batch_size = 64
    epsilon = 1.0
    eps_min = 0.1
    eps_decay = 0.995

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0

        while True:
            action = agent.act(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                agent.train_step(batch)

            if done:
                break

        agent.update_target()
        epsilon = max(eps_min, epsilon * eps_decay)
        print(f"Episode {episode}: reward = {total_reward}")