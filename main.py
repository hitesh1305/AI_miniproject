# main.py
import os, sys
sys.path.insert(0, os.path.dirname(__file__)) 
import pygame
import sys
from gui.main_menu import MainMenu



def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("AI Game Applications")
    clock = pygame.time.Clock()
    menu = MainMenu(screen)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            menu.handle_event(e)

        menu.update()
        menu.draw()
        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
