import pygame
import numpy as np
import random


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
        self.init_game()
        if self.render:
            self.draw_board()
        self.state = self.get_state()
        return self.state

    def set_snake(self):
        for s in self.snake:
            self.board[s[1]][s[0]] = 2
            if s == self.snake[0]:
                self.board[s[1]][s[0]] = 1

    def move(self, dx, dy):
        x, y = self.snake[0]
        new_x = x + dx
        new_y = y + dy
        if self.check_game_over(new_x, new_y):
            print("Game Over!")
            self.reward = -20
            self.done = True
            return

        for i in range(len(self.snake)):
            tmp_x, tmp_y = self.snake[i]
            self.board[tmp_y][tmp_x] = 0

        self.snake[0] = (new_x, new_y)
        for i in range(1, len(self.snake)):
            tmp_x, tmp_y = self.snake[i]
            self.snake[i] = (x, y)
            self.board[y][x] = 2
            x, y = tmp_x, tmp_y

        self.board[y][x] = 0
        if self.board[new_y][new_x] == 3:
            self.snake.append((x, y))
            self.board[y][x] = 2
            self.generate_apple(3)
            self.reward = 10
        elif self.board[new_y][new_x] == 4:
            rm_x, rm_y = self.snake[len(self.snake) - 1]
            self.snake.remove((rm_x, rm_y))
            self.board[rm_y][rm_x] = 0
            self.generate_apple(4)
            self.reward = -10
        elif self.board[new_y][new_x] == 0:
            self.reward = -0.01

        self.set_snake()

    def good_character(self, n):
        if n == 0:
            return "0"
        elif n == 1:
            return "H"
        elif n == 2:
            return "S"
        elif n == 3:
            return "G"
        elif n == 4:
            return "R"
        return "?"

    def print_state(self):
        x, y = self.snake[0]
        start_space = " " * (x + 1)
        end_space = " " * (self.size - x - 1)
        print("=" * (self.size + 2))
        print(start_space + "W" + end_space, flush=True)
        for i in range(self.size):
            if i != y:
                print(start_space + self.good_character(
                    self.board[i][x]) + end_space, flush=True)
                continue
            print("W", end="")
            for j in range(self.size):
                print(self.good_character(self.board[i][j]), end="")
            print("W", flush=True)
        print(start_space + "W" + end_space, flush=True)

    def step(self, action):
        self.done = False
        self.reward = 0
        if action == 0:
            self.move(0, -1)
        elif action == 1:
            self.move(0, 1)
        elif action == 2:
            self.move(-1, 0)
        elif action == 3:
            self.move(1, 0)
        self.steps += 1
        if not self.done:
            self.state = self.get_state()
            if self.render:
                self.draw_board()
        return self.state, self.reward, self.done

    def get_state(self):
        x, y = self.snake[0]

        obs = [0, 0, 0, 0]
        green = [0, 0, 0, 0]
        red = [0, 0, 0, 0]

        for d in range(1, y + 1):
            cell = self.board[y - d][x]
            if cell == 2:
                obs[0] = 1 / d
                break
            if cell == 3:
                green[0] = 1 / d
                break
            if cell == 4:
                red[0] = 1 / d
                break
        else:
            obs[0] = 1 / (y + 1)

        for d in range(1, self.size - y):
            cell = self.board[y + d][x]
            if cell == 2:
                obs[1] = 1 / d
                break
            if cell == 3:
                green[1] = 1 / d
                break
            if cell == 4:
                red[1] = 1 / d
                break
        else:
            obs[1] = 1 / (self.size - y)

        for d in range(1, x + 1):
            cell = self.board[y][x - d]
            if cell == 2:
                obs[2] = 1 / d
                break
            if cell == 3:
                green[2] = 1 / d
                break
            if cell == 4:
                red[2] = 1 / d
                break
        else:
            obs[2] = 1 / (x + 1)

        for d in range(1, self.size - x):
            cell = self.board[y][x + d]
            if cell == 2:
                obs[3] = 1 / d
                break
            if cell == 3:
                green[3] = 1 / d
                break
            if cell == 4:
                red[3] = 1 / d
                break
        else:
            obs[3] = 1 / (self.size - x)

        state = np.array([
            obs[0], obs[1], obs[2], obs[3],
            green[0], green[1], green[2], green[3],
            red[0], red[1], red[2], red[3]
        ], dtype=np.float32)

        return state

    def valid_cell(self, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.board[y][x] != 0:
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
        self.board[y][x] = 1
        for _ in range(2):
            tmp_x, tmp_y = self.generate_body(x, y)
            while not self.valid_cell(tmp_x, tmp_y):
                tmp_x, tmp_y = self.generate_body(x, y)
            x, y = tmp_x, tmp_y
            self.board[y][x] = 2
            self.snake.append((x, y))

    def generate_apple(self, type):
        x = random.randint(0, 9)
        y = random.randint(0, 9)
        while not self.valid_cell(x, y):
            x = random.randint(0, 9)
            y = random.randint(0, 9)
        self.board[y][x] = type

    def init_game(self):
        self.board = np.zeros((self.size, self.size))
        self.snake = []
        self.reward = 0
        self.steps = 0
        if self.render:
            self.screen = pygame.display.set_mode((self.size*self.scale,
                                                   self.size*self.scale))
            pygame.display.set_caption("SnAIke")
        self.generate_snake()
        self.generate_apple(3)
        self.generate_apple(3)
        self.generate_apple(4)

    def draw_board(self):
        colors = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        colors[self.board == 0] = [120, 180, 0]
        colors[self.board == 1] = [40, 0, 100]
        colors[self.board == 2] = [0, 50, 250]
        colors[self.board == 3] = [0, 255, 0]
        colors[self.board == 4] = [250, 0, 0]

        surface = pygame.surfarray.make_surface(colors.swapaxes(0, 1))
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
        if len(self.snake) == 1 and self.board[y][x] == 2:
            return True
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        if self.board[y][x] == 2:
            return True
        return False
