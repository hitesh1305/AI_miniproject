# gui/arcade_gui.py (patched run_arcade)
import os
import pygame
import torch
from environments.arcade_game import SimpleArcadeGame
from agents.dqn_agent import DQNAgent

WHITE=(255,255,255); BG=(20,20,30)
BLUE=(50,100,200); RED=(200,50,50); GREEN=(60,200,100); GRID=(40,40,60)
CELL_BG=(30,30,45); HUD=(220,220,220)

def run_arcade(screen):
    clock = pygame.time.Clock()
    W, H = screen.get_width(), screen.get_height()
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 28)
    title_font = pygame.font.Font(None, 64)

    env = SimpleArcadeGame(max_steps=None)
    agent = DQNAgent(env.state_size, len(env.get_valid_actions()))
    ckpt = os.path.join("checkpoints", "dqn_arcade_best.pth")
    if os.path.exists(ckpt):
        agent.load(ckpt, map_location=torch.device("cpu"))
        agent.epsilon = 0.0

    # View controls
    mode, paused, step_once = "ai", False, False
    fps_choices, fps_index = [5, 10, 15, 30, 60], 2  # 15 FPS default

    # Safe margins: leave space for title and 2 HUD lines
    side_margin = 40
    top_margin = 120
    bottom_margin = 88  # enough for two lines at 28–32px each

    # Dynamic cell size so grid fits between margins
    cell_w = (W - 2*side_margin) // env.W
    cell_h = (H - (top_margin + bottom_margin)) // env.H
    cell = max(10, min(cell_w, cell_h))  # keep cells visible at small windows

    gx = (W - env.W * cell) // 2
    gy = top_margin

    def draw_state():
        screen.fill(BG)
        title = title_font.render("Arcade (DQN)", True, WHITE)
        screen.blit(title, title.get_rect(center=(W//2, 40)))

        # grid border and cells
        pygame.draw.rect(screen, GRID, (gx-4, gy-4, env.W*cell+8, env.H*cell+8), 4)
        for r in range(env.H):
            for c in range(env.W):
                pygame.draw.rect(screen, CELL_BG, (gx + c*cell, gy + r*cell, cell-2, cell-2))

        # goal / obstacles / player
        pygame.draw.rect(screen, GREEN, (gx + env.goal_x*cell, gy + env.goal_y*cell, cell-2, cell-2))
        for (ox, oy) in env.obstacles:
            pygame.draw.rect(screen, RED, (gx + ox*cell, gy + oy*cell, cell-2, cell-2))
        pygame.draw.rect(screen, BLUE, (gx + env.player_x*cell, gy + (env.H-1)*cell, cell-2, cell-2))

    # main loop
    s = env.reset()
    total, steps, last_info = 0.0, 0, {}

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return "quit"
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return "menu"
                if e.key == pygame.K_r:
                    s = env.reset(); total = 0.0; steps = 0; last_info = {}
                if e.key == pygame.K_SPACE:
                    paused = not paused
                if e.key == pygame.K_n:
                    step_once = True
                if e.key == pygame.K_h:
                    mode = "human" if mode == "ai" else "ai"
                if e.key == pygame.K_f:
                    fps_index = (fps_index + 1) % len(fps_choices)
                # Human control
                if mode == "human" and not env.done:
                    if e.key == pygame.K_LEFT:
                        s, r, done, info = env.step(0); total += r; steps += 1; last_info = info
                    elif e.key == pygame.K_RIGHT:
                        s, r, done, info = env.step(2); total += r; steps += 1; last_info = info

        # AI step when not paused (or single-step when paused)
        if mode == "ai" and not env.done and ((not paused) or step_once):
            a = agent.act(s, env.get_valid_actions())
            s, r, done, info = env.step(a)
            total += r
            steps += 1
            last_info = info
            step_once = False

        draw_state()

        # Two-line HUD to avoid clipping
        hud1 = f"Mode:{mode.upper()}  |  Speed:{fps_choices[fps_index]} FPS  |  Steps:{steps}  |  Reward:{total:.2f}"
        hud2 = "Keys: H=toggle AI/Human, F=speed, SPACE=pause, N=step, R=restart, ESC=menu"

        # Measure and center within safe bottom area
        surf1 = font.render(hud1, True, HUD)
        rect1 = surf1.get_rect(center=(W//2, H - 48))   # line 1
        surf2 = small_font.render(hud2, True, HUD)
        rect2 = surf2.get_rect(center=(W//2, H - 20))   # line 2

        # Draw a subtle panel to improve readability
        panel_top = rect1.top - 6
        panel_height = (rect2.bottom + 6) - panel_top
        pygame.draw.rect(screen, (15,15,22), (0, panel_top, W, panel_height))

        screen.blit(surf1, rect1)
        screen.blit(surf2, rect2)

        # Terminal banner under title, never offscreen
        if env.done:
            tag = last_info.get("terminal")
            msg = "GOAL! +1" if tag == "goal" else "HIT! -1" if tag == "collision" else "ENDED"
            end_text = font.render(f"{msg}  |  R=restart  ESC=menu", True, WHITE)
            screen.blit(end_text, end_text.get_rect(center=(W//2, 90)))

        pygame.display.flip()
        clock.tick(fps_choices[fps_index])  # enforce view speed
