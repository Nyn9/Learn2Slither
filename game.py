import numpy as np
from environnement import Env
from agent import Agent


def main():
    env = Env(True)
    state_size = env.size + env.size
    agent = Agent(state_size)
    epochs = 1000
    batch_size = 32

    for epoch in range(epochs):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False

        print(f"Epoch: {epoch+1}")
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {},"
                      " e: {:.2}".format(epoch+1, epochs, reward, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)


if __name__ == "__main__":
    main()
