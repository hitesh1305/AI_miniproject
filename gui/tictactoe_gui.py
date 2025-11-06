# gui/tictactoe_gui.py
import pygame
import numpy as np
from environments.tictactoe import TicTacToe
from agents.minimax_agent import MinimaxAgent

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
BLUE, RED, GRAY = (50, 100, 200), (200, 50, 50), (200, 200, 200)

class TicTacToeGUI:
    def __init__(self, screen):
        self.screen = screen
        self.w, self.h = screen.get_width(), screen.get_height()
        self.env = TicTacToe()
        self.ai = MinimaxAgent(player=-1)  # Human is X (+1), AI is O (-1)
        self.cell = 120
        self.board_size = 3 * self.cell
        self.bx = (self.w - self.board_size) // 2
        self.by = 100
        self.font = pygame.font.Font(None, 48)
        self.title_font = pygame.font.Font(None, 64)
        self.message = "Your turn (X)"
        self.game_over = False
        self.env.reset()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over and self.env.current_player == 1:
            mx, my = event.pos
            col = (mx - self.bx) // self.cell
            row = (my - self.by) // self.cell
            if 0 <= row < 3 and 0 <= col < 3:
                move = (row, col)
                if move in self.env.get_valid_actions():
                    self._move(move)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self._reset()
            elif event.key == pygame.K_ESCAPE:
                return "menu"
        return None

    def _move(self, move):
        _, _, done, info = self.env.step(move)
        if done:
            self.game_over = True
            w = info.get("winner", 0)
            self.message = "You Win!" if w == 1 else ("AI Wins!" if w == -1 else "Draw!")
        else:
            self.message = "AI's turn (O)" if self.env.current_player == -1 else "Your turn (X)"

    def _reset(self):
        self.env.reset()
        self.game_over = False
        self.message = "Your turn (X)"

    def update(self):
        # AI plays when it's O's turn
        if not self.game_over and self.env.current_player == -1:
            valid = self.env.get_valid_actions()
            if valid:
                pygame.time.delay(250)  # small delay for UX
                move = self.ai.act(self.env.board, valid)
                self._move(move)

    def draw(self):
        self.screen.fill(WHITE)
        # title and message
        title = self.title_font.render("Tic-Tac-Toe", True, BLACK)
        self.screen.blit(title, title.get_rect(center=(self.w // 2, 40)))
        msg = self.font.render(self.message, True, BLUE)
        self.screen.blit(msg, msg.get_rect(center=(self.w // 2, self.by - 30)))

        # grid
        for i in range(4):
            x = self.bx + i * self.cell
            y = self.by + i * self.cell
            pygame.draw.line(self.screen, BLACK, (x, self.by), (x, self.by + self.board_size), 3)
            pygame.draw.line(self.screen, BLACK, (self.bx, y), (self.bx + self.board_size, y), 3)

        # pieces
        for r in range(3):
            for c in range(3):
                v = self.env.board[r, c]
                cx = self.bx + c * self.cell + self.cell // 2
                cy = self.by + r * self.cell + self.cell // 2
                if v == 1:
                    off = (self.cell - 40) // 2
                    pygame.draw.line(self.screen, BLUE, (cx - off, cy - off), (cx + off, cy + off), 8)
                    pygame.draw.line(self.screen, BLUE, (cx + off, cy - off), (cx - off, cy + off), 8)
                elif v == -1:
                    pygame.draw.circle(self.screen, RED, (cx, cy), self.cell // 2 - 20, 8)

        hint = pygame.font.Font(None, 24).render("R = restart | ESC = menu", True, GRAY)
        self.screen.blit(hint, hint.get_rect(center=(self.w // 2, self.h - 30)))

def run_tictactoe(screen):
    clock = pygame.time.Clock()
    game = TicTacToeGUI(screen)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return "quit"
            result = game.handle_event(e)
            if result == "menu":
                return "menu"
        game.update()
        game.draw()
        pygame.display.flip()
        clock.tick(60)
