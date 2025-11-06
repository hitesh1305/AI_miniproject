# gui/main_menu.py
import pygame
import sys

class Button:
    def __init__(self, x, y, w, h, text, color, hover_color):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.current_color = color
        self.font = pygame.font.Font(None, 36)

    def draw(self, surface):
        pygame.draw.rect(surface, self.current_color, self.rect, border_radius=10)
        pygame.draw.rect(surface, (255, 255, 255), self.rect, width=3, border_radius=10)
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.current_color = self.hover_color if self.rect.collidepoint(event.pos) else self.color
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return True
        return False

class MainMenu:
    def __init__(self, screen):
        self.screen = screen
        self.w, self.h = screen.get_width(), screen.get_height()
        self.bg = (30, 30, 50)
        self.title_font = pygame.font.Font(None, 72)
        self.sub_font = pygame.font.Font(None, 32)

        bw, bh = 300, 60
        bx = (self.w - bw) // 2
        sy, sp = 200, 80
        self.buttons = [
            Button(bx, sy + sp * 0, bw, bh, "Tic-Tac-Toe", (50, 100, 200), (70, 130, 230)),
            Button(bx, sy + sp * 1, bw, bh, "Connect Four", (200, 50, 100), (230, 70, 130)),
            Button(bx, sy + sp * 2, bw, bh, "Arcade Game", (100, 200, 50), (130, 230, 70)),
            Button(bx, sy + sp * 3, bw, bh, "Quit", (150, 50, 50), (200, 70, 70)),
        ]

    def handle_event(self, event):
        for i, b in enumerate(self.buttons):
            if b.handle_event(event):
                self._on_click(i)

    def _on_click(self, idx):
        if idx == 0:
            from gui.tictactoe_gui import run_tictactoe
            res = run_tictactoe(self.screen)
            if res == "quit":
                pygame.quit(); sys.exit()
        elif idx == 1:
            # Placeholder: will route to Connect 4 GUI in the next phase
            from gui.connect4_gui import run_connect4
            res = run_connect4(self.screen)
            if res == "quit":
                pygame.quit(); sys.exit()
        elif idx == 2:
            # Placeholder: will route to Arcade GUI (DQN) in the next phase
            from gui.arcade_gui import run_arcade
            res = run_arcade(self.screen)
            if res == "quit":
                pygame.quit(); sys.exit()
        elif idx == 3:
            pygame.quit(); sys.exit()

    def update(self):
        pass

    def draw(self):
        self.screen.fill(self.bg)
        title = self.title_font.render("AI Game Applications", True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(self.w // 2, 80)))
        subtitle = self.sub_font.render("Choose a game to play", True, (200, 200, 200))
        self.screen.blit(subtitle, subtitle.get_rect(center=(self.w // 2, 130)))
        for b in self.buttons:
            b.draw(self.screen)

