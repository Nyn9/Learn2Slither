import sys
import numpy as np
import pygame
from environnement import Env
from agent import Agent


def main():
    env = Env(True)
    state_size = env.size * env.size
    agent = Agent(state_size)
    epochs = 1000
    batch_size = 32

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    env.move(0, -1)
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    env.move(0, 1)
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a :
                    env.move(-1, 0)
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    env.move(1, 0)
        env.draw_board()

    # for epoch in range(epochs):
    #     state = env.reset()
    #     state = np.reshape(state, [1, state_size])
    #     done = False    

    #     print(f"Epoch: {epoch+1}")
    #     while not done:
    #         action = agent.choose_action(state)
    #         next_state, reward, done = env.step(action)
    #         next_state = np.reshape(next_state, [1, state_size])
    #         agent.remember(state, action, reward, next_state, done)
    #         state = next_state
    #         if done:
    #             print("episode: {}/{}, score: {},"
    #                   " e: {:.2}".format(epoch+1, epochs, reward, agent.epsilon))
    #             break
    #         if len(agent.memory) > batch_size:
    #             agent.replay(batch_size)


if __name__ == "__main__":
    main()
