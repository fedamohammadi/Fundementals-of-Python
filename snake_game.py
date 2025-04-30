import pygame
import random

# Initialize Pygame
pygame.init()

# Game constants
WIDTH = 400
HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE - 1  
BASE_FPS = 5
MAX_FPS = 15

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

# Snake and food
snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)] 
snake_dir = (1, 0)
food = (random.randint(0, GRID_WIDTH - 1), random.randint(1, GRID_HEIGHT - 1))  # Avoid top row
score = 0
running = True
game_over = False
current_fps = BASE_FPS
game_state = "start"
player_name = ""
input_active = True  


clock = pygame.time.Clock()


font = pygame.font.SysFont(None, 36)

def reset_game():
    global snake, snake_dir, food, score, game_over, current_fps
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    snake_dir = (1, 0)
    food = (random.randint(0, GRID_WIDTH - 1), random.randint(1, GRID_HEIGHT - 1))
    score = 0
    game_over = False
    current_fps = BASE_FPS

def draw_button(text, x, y, width, height, inactive_color, active_color):
    mouse = pygame.mouse.get_pos()
    button_rect = pygame.Rect(x, y, width, height)
    color = active_color if button_rect.collidepoint(mouse) else inactive_color
    pygame.draw.rect(screen, color, button_rect)
    text_surf = font.render(text, True, WHITE)
    text_rect = text_surf.get_rect(center=button_rect.center)
    screen.blit(text_surf, text_rect)
    return button_rect

while running:
    # Handle events
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
                elif event.unicode.isprintable() and len(player_name) < 20:
                    player_name += event.unicode
            elif game_state == "playing" and not game_over:
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
    
    if game_state == "playing" and not game_over:
        # Update FPS based on score
        current_fps = min(BASE_FPS + (score // 10), MAX_FPS)
        
        # Move snake
        head = (snake[0][0] + snake_dir[0], snake[0][1] + snake_dir[1])
        
        # Check for collisions
        if (head[0] < 0 or head[0] >= GRID_WIDTH or 
            head[1] < 1 or head[1] >= GRID_HEIGHT or  # Avoid top row
            head in snake):
            game_over = True
            continue
        
        snake.insert(0, head)
        
        # Check if food is eaten
        if head == food:
            score += 1
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(1, GRID_HEIGHT - 1))
        else:
            snake.pop()
    
    # Draw
    screen.fill(BLACK)
    
    if game_state == "start":
        prompt_text = font.render("What is your name?", True, WHITE)
        prompt_rect = prompt_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
        screen.blit(prompt_text, prompt_rect)
        
        input_text = font.render(player_name + ("|" if input_active else ""), True, WHITE)
        input_rect = input_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 25))
        pygame.draw.rect(screen, GRAY, (WIDTH // 2 - 100, HEIGHT // 2, 200, 50))
        screen.blit(input_text, input_rect)
        
        submit_button = draw_button("Submit", WIDTH // 2 - 100, HEIGHT // 2 + 80, 200, 50, GRAY, GREEN)
        
    elif game_state == "welcome":
        welcome_text = font.render(f"Hi {player_name}! Welcome to Snake Game", True, WHITE)
        welcome_rect = welcome_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
        screen.blit(welcome_text, welcome_rect)
        
        start_button = draw_button("Start", WIDTH // 2 - 100, HEIGHT // 2, 200, 50, GRAY, GREEN)
        
    elif game_state == "playing":
        if game_over:
            # Draw game over screen
            game_over_text = font.render(f"Game Over! Score: {score}", True, WHITE)
            game_over_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
            screen.blit(game_over_text, game_over_rect)
            
            # Draw buttons
            restart_button = draw_button("Restart", WIDTH // 2 - 110, HEIGHT // 2, 100, 50, GRAY, GREEN)
            exit_button = draw_button("Exit", WIDTH // 2 + 10, HEIGHT // 2, 100, 50, GRAY, RED)
        else:
            # Draw snake and food
            for segment in snake:
                pygame.draw.rect(screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(screen, RED, (food[0] * GRID_SIZE, food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            
            # Draw score
            score_text = font.render(f'Score: {score}', True, WHITE)
            score_rect = score_text.get_rect(center=(WIDTH // 2, 10))
            screen.blit(score_text, score_rect)
    
    pygame.display.flip()
    if game_state == "playing":
        clock.tick(current_fps)
    else:
        clock.tick(30)  # Faster refresh for input screens

# Cleanup
pygame.quit()