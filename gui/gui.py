import pygame
import sys
import os
import numpy as np
from game.game import PokerGame
from game.player import Player
from game.card import Card

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Heads-Up Poker AI")

# Load card images
def load_card_images():
    card_images = {}
    suits = ['hearts', 'diamonds', 'clubs', 'spades']
    ranks = [str(n) for n in range(2, 11)] + ['J', 'Q', 'K', 'A']
    for suit in suits:
        for rank in ranks:
            card_name = f"{suit}_{rank}"
            image_path = os.path.join('assets', 'cards', f"{card_name}.png")
            if os.path.exists(image_path):
                image = pygame.image.load(image_path)
                card_images[card_name] = image
            else:
                print(f"Warning: Image {image_path} not found.")
    return card_images

CARD_IMAGES = load_card_images()

# Main GUI class
class PokerGUI:
    def __init__(self):
        self.game = PokerGame(
            Player('Human', 1000),
            Player('AI', 1000)
        )
        self.font = pygame.font.SysFont(None, 24)

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        img = self.font.render(text, True, color)
        SCREEN.blit(img, (x, y))

    def draw_card(self, card, x, y):
        card_name = f"{card.rank}_of_{card.suit}"
        image = CARD_IMAGES.get(card_name)
        if image:
            SCREEN.blit(image, (x, y))
        else:
            # Draw a placeholder rectangle if image not found
            pygame.draw.rect(SCREEN, (255, 0, 0), (x, y, 71, 96))  # Standard card size

    def draw_hand(self, hand, x, y):
        for idx, card in enumerate(hand.cards):
            self.draw_card(card, x + idx * 75, y)  # 75 pixels apart

    def main_loop(self):
        running = True
        while running:
            SCREEN.fill((0, 128, 0))  # Green background for poker table

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw human player's hand
            self.draw_text("Your Hand:", 50, SCREEN_HEIGHT - 150)
            self.draw_hand(self.game.players[0].hand, 50, SCREEN_HEIGHT - 130)

            # Draw AI player's hand (face down)
            self.draw_text("AI's Hand:", 50, 50)
            for idx in range(2):
                back_image = pygame.image.load(os.path.join('assets', 'cards', 'back.png'))
                SCREEN.blit(back_image, (50 + idx * 75, 70))

            # Draw community cards
            self.draw_text("Community Cards:", SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50)
            self.draw_hand(self.game.community_cards, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 30)

            # Update display
            pygame.display.flip()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    gui = PokerGUI()
    gui.game.play_round()  # Start a game round
    gui.main_loop()
