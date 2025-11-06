# gui/connect4_gui.py
import pygame
import numpy as np
from environments.connect_four import ConnectFour
from agents.mcts_agent import MCTSAgent

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 200)   # X
RED = (200, 50, 50)     # O
BG = (25, 25, 40)
GRAY = (200, 200, 200)
YELLOW = (240, 210, 70)  # hover highlight

class Connect4GUI:
    def __init__(self, screen, sims=800, c=1.41):
        self.screen = screen
        self.w, self.h = screen.get_width(), screen.get_height()
        self.env = ConnectFour()
        self.env.reset()
        # Human plays X (+1), AI plays O (-1)
        self.human = 1
        self.ai_player = -1
        self.ai = MCTSAgent(player=self.ai_player, num_simulations=sims, c=c)
        # Layout
        self.cols = 7
        self.rows = 6
        self.cell = 80
        self.board_w = self.cols * self.cell
        self.board_h = self.rows * self.cell
        self.bx = (self.w - self.board_w) // 2
        self.by = 100
        self.title_font = pygame.font.Font(None, 64)
        self.msg_font = pygame.font.Font(None, 40)
        self.small_font = pygame.font.Font(None, 24)
        self.message = "Your turn (X)"
        self.game_over = False
        self.hover_col = None

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.by <= my <= self.by + self.board_h:
                self.hover_col = (mx - self.bx) // self.cell
                if not (0 <= self.hover_col < self.cols):
                    self.hover_col = None
            else:
                self.hover_col = None

        if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over and self.env.current_player == self.human:
            if self.hover_col is not None and self.hover_col in self.env.get_valid_actions():
                self._move(self.hover_col)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self._reset()
            elif event.key == pygame.K_ESCAPE:
                return "menu"
        return None

    def _move(self, col):
        _, _, done, info = self.env.step(col)
        if done:
            self.game_over = True
            w = info.get("winner", 0)
            self.message = "You Win!" if w == self.human else ("AI Wins!" if w == self.ai_player else "Draw!")
        else:
            self.message = "AI thinking..." if self.env.current_player == self.ai_player else "Your turn (X)"

    def _reset(self):
        self.env.reset()
        self.game_over = False
        self.message = "Your turn (X)"

    def update(self):
        if not self.game_over and self.env.current_player == self.ai_player:
            valid = self.env.get_valid_actions()
            if valid:
                # brief visual delay for UX
                pygame.time.delay(200)
                action = self.ai.act(self.env.board, valid)
                self._move(action)

    def draw(self):
        self.screen.fill(BG)
        title = self.title_font.render("Connect Four (MCTS)", True, WHITE)
        self.screen.blit(title, title.get_rect(center=(self.w // 2, 40)))
        msg = self.msg_font.render(self.message, True, BLUE if "turn" in self.message else WHITE)
        self.screen.blit(msg, msg.get_rect(center=(self.w // 2, self.by - 30)))

        # draw grid background
        pygame.draw.rect(self.screen, BLACK, (self.bx - 4, self.by - 4, self.board_w + 8, self.board_h + 8), 4)
        for r in range(self.rows):
            for c in range(self.cols):
                cx = self.bx + c * self.cell + self.cell // 2
                cy = self.by + r * self.cell + self.cell // 2
                pygame.draw.circle(self.screen, (30, 30, 60), (cx, cy), self.cell // 2 - 6)

        # highlight hover column
        if self.hover_col is not None and not self.game_over and self.env.current_player == self.human:
            x0 = self.bx + self.hover_col * self.cell
            pygame.draw.rect(self.screen, YELLOW, (x0, self.by, self.cell, self.board_h), 4)

        # draw pieces
        for r in range(self.rows):
            for c in range(self.cols):
                v = self.env.board[r, c]
                if v != 0:
                    cx = self.bx + c * self.cell + self.cell // 2
                    cy = self.by + r * self.cell + self.cell // 2
                    color = BLUE if v == 1 else RED
                    pygame.draw.circle(self.screen, color, (cx, cy), self.cell // 2 - 10)

        hint = self.small_font.render("R = restart | ESC = menu", True, GRAY)
        self.screen.blit(hint, hint.get_rect(center=(self.w // 2, self.h - 30)))

def run_connect4(screen):
    clock = pygame.time.Clock()
    game = Connect4GUI(screen)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return "quit"
            res = game.handle_event(e)
            if res == "menu":
                return "menu"
        game.update()
        game.draw()
        pygame.display.flip()
        clock.tick(60)
