import numpy as np
import random
import pygame
import time



class Env:
    def __init__(self):
        pygame.init()
        self.size = 10
        self.scale = 90
        self.board = np.zeros((self.size, self.size))
        self.snake = []
        self.init_game()
        self.draw_board()
        self.state = self.snake[0]


    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.snake = []
        self.init_game()
        self.draw_board()
        self.state = self.snake[0]
        return self.state


    def move(self, dx, dy):
        x, y = self.snake[0]
        new_x = x + dx
        new_y = y + dy
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            print("Game Over!")
            self.reward = -100
            self.done = True
            return

        self.snake[0] = (new_x, new_y)
        for i in range(1, len(self.snake)):
            tmp_x, tmp_y = self.snake[i]
            self.snake[i] = (x, y)
            x, y = tmp_x, tmp_y


        self.board[x][y] = 0
        if self.board[new_x][new_y] == 3:
            self.snake.append((x, y))
            self.board[x][y] = 2
            # self.generate_apple(3)
            self.reward = 10
        elif self.board[new_x][new_y] == 4:
            rm_x, rm_y = self.snake[len(self.snake) - 1]
            self.snake.remove((rm_x, rm_y))
            self.board[rm_x][rm_y] = 0
            # self.generate_apple(4)
            self.reward = -10
        elif self.board[new_x][new_y] == 0:
            self.reward = -1

        if self.check_game_over(new_x, new_y):
            print("Game Over!")
            self.reward = -100
            self.done = True
            return


    def step(self, action):
        self.reward = 0
        self.done = False
        if action == 0:
            self.move(0, -1)
        elif action == 1:
            self.move(0, 1)
        elif action == 2:
            self.move(-1, 0)
        elif action == 3:
            self.move(1, 0)
        if len(self.snake) != 0:
            self.state = self.snake[0]
            self.draw_board()
        return self.state, self.reward, self.done


    def get_state(self):
        x, y = self.snake[0]
        state = []
        for i in range(10):
            state.append(self.board[x][i])
            state.append(self.board[i][y])

        return self.snake[0]

    def valid_cell(self, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.board[x][y] != 0:
            return False
        return True


    def generate_body(self, x, y):
        next_cell = random.choice([-1, 1])
        dir_cell = random.choice([0, 1])
        if dir_cell == 0:
            x += next_cell
        else:
            y += next_cell
        return x, y


    def generate_snake(self):
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        self.snake.append((x, y))
        self.board[x][y] = 1
        for _ in range(2):
            tmp_x, tmp_y = self.generate_body(x, y)
            while not self.valid_cell(tmp_x, tmp_y):
                tmp_x, tmp_y = self.generate_body(x, y)
            x, y = tmp_x, tmp_y
            self.board[x][y] = 2
            self.snake.append((x, y))


    def generate_apple(self, type):
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        while not self.valid_cell(x, y):
            x = random.randint(0, 9)
            y = random.randint(0, 9)
        self.board[x][y] = type


    def init_game(self):
        self.screen = pygame.display.set_mode((self.size*self.scale, self.size*self.scale))
        self.generate_snake()
        self.board[3][6] = 3
        self.board[5][5] = 3
        self.board[7][2] = 4
        # self.generate_apple(3)
        # self.generate_apple(3)
        # self.generate_apple(4)


    def draw_board(self):
        for s in self.snake:
            self.board[s[0]][s[1]] = 2
            if s == self.snake[0]:
                self.board[s[0]][s[1]] = 1
        colors = np.zeros((self.size, self.size, 3), dtype=int)
        colors[self.board == 0] = [120, 180, 0]
        colors[self.board == 1] = [40, 0, 100]
        colors[self.board == 2] = [0, 50, 250]
        colors[self.board == 3] = [0, 255, 0]
        colors[self.board == 4] = [250, 0, 0]

        surface = pygame.surfarray.make_surface(colors)
        surface = pygame.transform.scale(surface, (self.size*self.scale, self.size*self.scale))
        self.screen.blit(surface, (0, 0))
        for i in range(self.size + 1):
            pygame.draw.line(self.screen, (0, 0, 0), (i*self.scale, 0),
                            (i*self.scale, self.size*self.scale))
            pygame.draw.line(self.screen, (0, 0, 0), (0, i*self.scale),
                            (self.size*self.scale, i*self.scale))

        pygame.display.flip()


    def check_game_over(self, x, y):
        if len(self.snake) == 0:
            return True
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        if self.board[x][y] == 2:
            return True
        return False


class Agent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.5):
        self.q_table = np.zeros((10, 10, 4))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])
        # print(f"State: {state}")
        cur_q = self.q_table[state][action]
        self.q_table[state][action] = cur_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - cur_q
        )
        print(self.q_table)


def main():
    env = Env()
    agent = Agent()
    epochs = 1000

    for epoch in range(epochs):
        state = env.reset()
        done = False

        print(f"Epoch: {epoch+1}")
        while not done:
            time.sleep(0.05)
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state



if __name__ == "__main__":
    main()
