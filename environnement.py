import pygame
import numpy as np
import random
from time import sleep


class Env:
    def __init__(self, render=False):
        self.render = render
        if self.render:
            pygame.init()
        self.size = 10
        self.scale = 90
        self.init_game()
        if self.render:
            self.draw_board()
        self.state = self.get_state()

    def reset(self):
        print("=========")
        self.init_game()
        if self.render:
            self.draw_board()
        self.state = self.get_state()
        return self.state

    def move(self, dx, dy):
        print(f"Move: dx={dx}, dy={dy}")
        x, y = self.snake[0]
        new_x = x + dx
        new_y = y + dy
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            print("Game Over!")
            self.reward -= 200
            self.done = True
            return

        print(self.board)
        print(self.snake)

        for j in range(10):
            for i in range(10):
                print(f"x : {i} / y : {j} : {self.board[j][i]} ")

        for i in range(len(self.snake)):
            tmp_x, tmp_y = self.snake[i]
            self.board[tmp_y][tmp_x] = 0

        self.snake[0] = (new_x, new_y)
        self.board[new_y][new_x] = 1
        print(f"Snake head : {self.snake[0]}")
        x, y = new_x, new_y
        for i in range(1, len(self.snake)):
            tmp_x, tmp_y = self.snake[i]
            self.snake[i] = (x, y)
            print(f"Snake body {i} : {self.snake[i]}")
            self.board[y][x] = 2
            x, y = tmp_x, tmp_y

        print("After moving:")
        print(self.snake)

        self.board[y][x] = 0
        if self.board[new_y][new_x] == 3:
            self.snake.append((x, y))
            self.board[y][x] = 2
            self.generate_apple(3)
            self.reward += 100
        elif self.board[new_y][new_x] == 4:
            rm_x, rm_y = self.snake[len(self.snake) - 1]
            self.snake.remove((rm_x, rm_y))
            self.board[rm_y][rm_x] = 0
            self.generate_apple(4)
            self.reward -= 100
        elif self.board[new_y][new_x] == 0:
            self.reward -= 10

        if self.check_game_over(new_x, new_y):
            print("Game Over!")
            self.reward -= 200
            self.done = True
            return

    def step(self, action):
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
            self.state = self.get_state()
            if self.render:
                self.draw_board()
        return self.state, self.reward, self.done

    def get_state(self):
        x, y = self.snake[0]
        print(f"Snake Head Position: ({x}, {y})")
        state = np.zeros((10, 10))
        for i in range(10):
            state[y][i] = self.board[y][i]
            state[i][x] = self.board[i][x]

        print("Board State:")
        print(self.board)
        print("Current State:")
        print(state)

        return state

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
        print("Initial Snake Position:")
        print(f"Snake head : {self.snake[0]}")
        for i in range(1, len(self.snake)):
            print(f"Snake body {i} : {self.snake[i]}")
        print("------")

    def generate_apple(self, type):
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        while not self.valid_cell(x, y):
            x = random.randint(0, 9)
            y = random.randint(0, 9)
        self.board[x][y] = type

    def init_game(self):
        self.board = np.zeros((self.size, self.size))
        self.snake = []
        self.reward = 0
        if self.render:
            self.screen = pygame.display.set_mode((self.size*self.scale,
                                                   self.size*self.scale))
            pygame.display.set_caption("SnAIke")
        self.generate_snake()
        self.generate_apple(3)
        self.generate_apple(3)
        self.generate_apple(4)

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
        surface = pygame.transform.scale(surface, (self.size*self.scale,
                                                   self.size*self.scale))
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
