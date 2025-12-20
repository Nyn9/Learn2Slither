import numpy as np
from environnement import Env
from agent import Agent
import matplotlib.pyplot as plt


def main():
    env = Env()
    state_size = 12
    agent = Agent(state_size)
    epochs = 1000
    batch_size = 32
    rewards = []
    sizes = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    for epoch in range(epochs):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        reward_per_epoch = 0

        print(f"Epoch: {epoch+1}")
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            reward_per_epoch += reward
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if done:
                if (epoch + 1) == 1 or (epoch + 1) == 50 or (epoch + 1) % 100 == 0:
                    name = "models/snaike_" + str(epoch + 1) + ".weights.h5"
                    agent.save(name)
                rewards.append(reward_per_epoch)
                sizes.append(len(env.snake))
                ax1.plot(rewards, color='blue')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Reward')
                ax2.plot(sizes, color='red')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Snake Size')
                plt.pause(0.01)
                print("episode: {}/{}, score: {}, size: {},"
                      " e: {:.2}".format(epoch+1, epochs, reward_per_epoch, len(env.snake), agent.epsilon))


if __name__ == "__main__":
    main()
