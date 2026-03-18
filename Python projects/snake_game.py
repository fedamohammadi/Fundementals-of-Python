# Snake Game (Python + Pygame)
'''
This is a simple Snake game built using Python and Pygame. It is one of my first small projects 
to practice programming concepts in a more practical and interactive way.

The game includes basic features such as movement, collision detection, score tracking, and a 
dynamic speed system where the snake gradually becomes faster as it eats more food. 

'''


import random
import pygame

# This initializes the Pygame ------

pygame.init()

# ===============================
# Game Constants
# ===============================

WIDTH = 1000
HEIGHT = 700
GRID_SIZE = 20

PLAY_TOP_MARGIN = 60
PLAY_WIDTH = WIDTH
PLAY_HEIGHT = HEIGHT - PLAY_TOP_MARGIN

GRID_WIDTH = PLAY_WIDTH // GRID_SIZE
GRID_HEIGHT = PLAY_HEIGHT // GRID_SIZE

BASE_FPS = 4
MAX_FPS = 12
FPS_INCREASE_EVERY = 1  # Increase speed every food eaten

NAME_LIMIT = 20

# ===============================
# Colors
# ===============================

BLACK = (18, 18, 18)
WHITE = (245, 245, 245)
RED = (220, 50, 50)
GREEN = (34, 177, 76)
DARK_GREEN = (20, 110, 50)
LIGHT_GREEN = (120, 220, 120)
GRAY = (110, 110, 110)
DARK_GRAY = (55, 55, 55)
GOLD = (255, 210, 70)

# ===============================
# Screen and Fonts
# ===============================

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

title_font = pygame.font.SysFont(None, 56)
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 28)

clock = pygame.time.Clock()

# ===============================
# Game State
# ===============================

snake = []
snake_dir = (1, 0)
food = (0, 0)

score = 0
running = True
game_over = False
current_fps = BASE_FPS

game_state = "start"   # start, welcome, playing
player_name = ""
input_active = True

submit_button = pygame.Rect(0, 0, 0, 0)
start_button = pygame.Rect(0, 0, 0, 0)
restart_button = pygame.Rect(0, 0, 0, 0)
exit_button = pygame.Rect(0, 0, 0, 0)

# ===============================
# Helper Functions
# ===============================

def get_random_food(snake_body):
    """Return a random food position that does not overlap the snake."""
    while True:
        position = (
            random.randint(0, GRID_WIDTH - 1),
            random.randint(0, GRID_HEIGHT - 1),
        )
        if position not in snake_body:
            return position


def reset_game():
    """Reset the game state for a new round."""
    global snake, snake_dir, food, score, game_over, current_fps

    center_x = GRID_WIDTH // 2
    center_y = GRID_HEIGHT // 2

    # It start with a longer snake so it looks more like a snake immediately.
    snake = [
        (center_x, center_y),
        (center_x - 1, center_y),
        (center_x - 2, center_y),
    ]
    snake_dir = (1, 0)
    food = get_random_food(snake)
    score = 0
    game_over = False
    current_fps = BASE_FPS


def draw_button(text, x, y, width, height, inactive_color, active_color):
    """Draw a hoverable button and return its rectangle."""
    mouse = pygame.mouse.get_pos()
    button_rect = pygame.Rect(x, y, width, height)
    color = active_color if button_rect.collidepoint(mouse) else inactive_color

    pygame.draw.rect(screen, color, button_rect, border_radius=12)
    pygame.draw.rect(screen, WHITE, button_rect, width=2, border_radius=12)

    text_surf = font.render(text, True, WHITE)
    text_rect = text_surf.get_rect(center=button_rect.center)
    screen.blit(text_surf, text_rect)

    return button_rect


def draw_snake():
    """Draw the snake with rounded segments so it looks less blocky."""
    for i, segment in enumerate(snake):
        x = segment[0] * GRID_SIZE
        y = PLAY_TOP_MARGIN + segment[1] * GRID_SIZE
        rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)

        if i == 0:
            # Head
            pygame.draw.rect(screen, LIGHT_GREEN, rect, border_radius=8)

            eye_radius = 2
            if snake_dir == (1, 0):   # right
                eye1 = (x + 14, y + 6)
                eye2 = (x + 14, y + 14)
            elif snake_dir == (-1, 0):  # left
                eye1 = (x + 6, y + 6)
                eye2 = (x + 6, y + 14)
            elif snake_dir == (0, -1):  # up
                eye1 = (x + 6, y + 6)
                eye2 = (x + 14, y + 6)
            else:  # down
                eye1 = (x + 6, y + 14)
                eye2 = (x + 14, y + 14)

            pygame.draw.circle(screen, BLACK, eye1, eye_radius)
            pygame.draw.circle(screen, BLACK, eye2, eye_radius)
        else:
            # Body
            pygame.draw.rect(screen, GREEN, rect, border_radius=7)
            inner_rect = rect.inflate(-6, -6)
            pygame.draw.rect(screen, DARK_GREEN, inner_rect, border_radius=5)


def draw_food():
    """Draw food as a small apple-like circle."""
    x = food[0] * GRID_SIZE + GRID_SIZE // 2
    y = PLAY_TOP_MARGIN + food[1] * GRID_SIZE + GRID_SIZE // 2

    pygame.draw.circle(screen, RED, (x, y), GRID_SIZE // 2 - 2)
    pygame.draw.rect(screen, GREEN, (x - 2, y - 12, 4, 6), border_radius=2)


def draw_score_bar():
    """Draw the score and speed at the top of the screen."""
    bar_rect = pygame.Rect(0, 0, WIDTH, PLAY_TOP_MARGIN)
    pygame.draw.rect(screen, DARK_GRAY, bar_rect)

    score_text = font.render(f"Score: {score}", True, WHITE)
    speed_text = small_font.render(f"Speed: {current_fps}", True, GOLD)

    screen.blit(score_text, (20, 15))
    screen.blit(speed_text, (170, 20))


# Initial Setup -----------

reset_game()

# ===============================
# Main Game Loop
# ===============================

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            if game_state == "start" and input_active:
                if event.key == pygame.K_RETURN and player_name.strip():
                    game_state = "welcome"
                elif event.key == pygame.K_BACKSPACE:
                    player_name = player_name[:-1]
                elif event.unicode.isprintable() and len(player_name) < NAME_LIMIT:
                    player_name += event.unicode

            elif game_state == "playing" and not game_over:
                # Prevent immediate reversal into the snake's body.
                if event.key == pygame.K_UP and snake_dir != (0, 1):
                    snake_dir = (0, -1)
                elif event.key == pygame.K_DOWN and snake_dir != (0, -1):
                    snake_dir = (0, 1)
                elif event.key == pygame.K_LEFT and snake_dir != (1, 0):
                    snake_dir = (-1, 0)
                elif event.key == pygame.K_RIGHT and snake_dir != (-1, 0):
                    snake_dir = (1, 0)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos

            if game_state == "start" and submit_button.collidepoint(mouse_pos) and player_name.strip():
                game_state = "welcome"

            elif game_state == "welcome" and start_button.collidepoint(mouse_pos):
                game_state = "playing"

            elif game_state == "playing" and game_over:
                if restart_button.collidepoint(mouse_pos):
                    reset_game()
                elif exit_button.collidepoint(mouse_pos):
                    running = False

    # ===============================
    # Here's the Game Logic
    # ===============================

    if game_state == "playing" and not game_over:
        # Snake gets faster every time it eats food.
        current_fps = min(BASE_FPS + (score // 5), MAX_FPS)

        head_x, head_y = snake[0]
        dx, dy = snake_dir
        new_head = (head_x + dx, head_y + dy)

        # Wall collision
        if (
            new_head[0] < 0
            or new_head[0] >= GRID_WIDTH
            or new_head[1] < 0
            or new_head[1] >= GRID_HEIGHT
        ):
            game_over = True

        # Self collision
        elif new_head in snake:
            game_over = True

        if not game_over:
            snake.insert(0, new_head)

            if new_head == food:
                score += 1
                food = get_random_food(snake)
            else:
                snake.pop()

    # ===============================
    # Drawing everything on the screen
    # ===============================

    screen.fill(BLACK)

    if game_state == "start":
        title_text = title_font.render("Snake Game", True, LIGHT_GREEN)
        title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 120))
        screen.blit(title_text, title_rect)

        prompt_text = font.render("What is your name?", True, WHITE)
        prompt_rect = prompt_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40))
        screen.blit(prompt_text, prompt_rect)

        input_box = pygame.Rect(WIDTH // 2 - 180, HEIGHT // 2, 360, 55)
        pygame.draw.rect(screen, DARK_GRAY, input_box, border_radius=12)
        pygame.draw.rect(screen, WHITE, input_box, width=2, border_radius=12)

        input_text = font.render(player_name + ("|" if input_active else ""), True, WHITE)
        input_rect = input_text.get_rect(center=input_box.center)
        screen.blit(input_text, input_rect)

        submit_button = draw_button(
            "Submit",
            WIDTH // 2 - 110,
            HEIGHT // 2 + 95,
            220,
            55,
            GRAY,
            GREEN,
        )

    elif game_state == "welcome":
        welcome_text = title_font.render(f"Hi {player_name}!", True, LIGHT_GREEN)
        welcome_rect = welcome_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60))
        screen.blit(welcome_text, welcome_rect)

        subtitle_text = font.render("Welcome to Snake Game", True, WHITE)
        subtitle_rect = subtitle_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(subtitle_text, subtitle_rect)

        start_button = draw_button(
            "Start",
            WIDTH // 2 - 110,
            HEIGHT // 2 + 70,
            220,
            55,
            GRAY,
            GREEN,
        )

    elif game_state == "playing":
        draw_score_bar()

        if game_over:
            game_over_text = title_font.render("Game Over", True, RED)
            game_over_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 80))
            screen.blit(game_over_text, game_over_rect)

            final_score_text = font.render(f"Final Score: {score}", True, WHITE)
            final_score_rect = final_score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 25))
            screen.blit(final_score_text, final_score_rect)

            restart_button = draw_button(
                "Restart",
                WIDTH // 2 - 130,
                HEIGHT // 2 + 40,
                120,
                55,
                GRAY,
                GREEN,
            )
            exit_button = draw_button(
                "Exit",
                WIDTH // 2 + 10,
                HEIGHT // 2 + 40,
                120,
                55,
                GRAY,
                RED,
            )
        else:
            draw_food()
            draw_snake()

    pygame.display.flip()

    if game_state == "playing":
        clock.tick(current_fps)
    else:
        clock.tick(30)


# Cleanup -----------------------
pygame.quit()