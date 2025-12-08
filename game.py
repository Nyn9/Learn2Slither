import numpy as np
import random
import pygame


size = 10
scale = 90
board = np.zeros((size, size))
snake = []


def valid_cell(x, y):
    if x < 0 or x >= size or y < 0 or y >= size:
        return False
    if board[x][y] != 0:
        return False
    return True


def generate_body(x, y):
    next_cell = random.choice([-1, 1])
    dir_cell = random.choice([0, 1])
    if dir_cell == 0:
        x += next_cell
    else:
        y += next_cell
    return x, y


def generate_snake():
    x = random.randint(0, 9)
    y = random.randint(0, 9)
    snake.append((x, y))
    board[x][y] = 1
    for _ in range(2):
        tmp_x, tmp_y = generate_body(x, y)
        while not valid_cell(tmp_x, tmp_y):
            tmp_x, tmp_y = generate_body(x, y)
        x, y = tmp_x, tmp_y
        board[x][y] = 2
        snake.append((x, y))


def generate_apple(type):
    x = random.randint(0, 9)
    y = random.randint(0, 9)
    while not valid_cell(x, y):
        x = random.randint(0, 9)
        y = random.randint(0, 9)
    board[x][y] = type


def init_game():
    pygame.init()
    screen = pygame.display.set_mode((size*scale, size*scale))
    generate_snake()
    generate_apple(3)
    generate_apple(3)
    generate_apple(4)
    return screen


def draw_board(screen):
    for s in snake:
        board[s[0]][s[1]] = 2
        if s == snake[0]:
            board[s[0]][s[1]] = 1
    colors = np.zeros((size, size, 3), dtype=int)
    colors[board == 0] = [120, 180, 0]
    colors[board == 1] = [40, 0, 100]
    colors[board == 2] = [0, 50, 250]
    colors[board == 3] = [0, 255, 0]
    colors[board == 4] = [250, 0, 0]

    surface = pygame.surfarray.make_surface(colors)
    surface = pygame.transform.scale(surface, (size*scale, size*scale))
    screen.blit(surface, (0, 0))

    for i in range(size + 1):
        pygame.draw.line(screen, (0, 0, 0), (i*scale, 0),
                         (i*scale, size*scale))
        pygame.draw.line(screen, (0, 0, 0), (0, i*scale),
                         (size*scale, i*scale))

    pygame.display.flip()


def check_game_over(x, y):
    if len(snake) == 0:
        return True
    if x < 0 or x >= size or y < 0 or y >= size:
        return True
    if board[x][y] == 2:
        return True
    return False


def move(dx, dy):
    x, y = snake[0]
    new_x = x + dx
    new_y = y + dy
    snake[0] = (new_x, new_y)
    for i in range(1, len(snake)):
        tmp_x, tmp_y = snake[i]
        snake[i] = (x, y)
        x, y = tmp_x, tmp_y

    if new_x < 0 or new_x >= size or new_y < 0 or new_y >= size:
        print("Game Over!")
        pygame.quit()
        exit()

    board[x][y] = 0
    if board[new_x][new_y] == 3:
        snake.append((x, y))
        board[x][y] = 2
        generate_apple(3)
    elif board[new_x][new_y] == 4:
        rm_x, rm_y = snake[len(snake) - 1]
        snake.remove((rm_x, rm_y))
        board[rm_x][rm_y] = 0
        generate_apple(4)

    if check_game_over(new_x, new_y):
        print("Game Over!")
        pygame.quit()
        exit()


def game_loop(screen):
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_w:
                    move(0, -1)
                if event.key == pygame.K_s:
                    move(0, 1)
                if event.key == pygame.K_a:
                    move(-1, 0)
                if event.key == pygame.K_d:
                    move(1, 0)
        draw_board(screen)
    pygame.quit()


def main():
    screen = init_game()
    game_loop(screen)


if __name__ == "__main__":
    main()
