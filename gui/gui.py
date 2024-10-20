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
    
    def draw_button(self, text, x, y, width, height, action=None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        button_rect = pygame.Rect(x, y, width, height)

        if button_rect.collidepoint(mouse):
            color = (170, 170, 170)  # Hover color
            if click[0] == 1 and action is not None:
                action()
        else:
            color = (100, 100, 100)  # Normal color

        pygame.draw.rect(SCREEN, color, button_rect)
        self.draw_text(text, x + 10, y + 10)


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


    def bet_action(self):
        # Implement betting logic
        amount = 10  # Placeholder amount
        self.game.players[0].bet(amount)
        self.game.pot += amount
        print(f"Human bets {amount}.")

    def call_action(self):
        # Implement call logic
        amount = 10  # Placeholder amount
        self.game.players[0].bet(amount)
        self.game.pot += amount
        print(f"Human calls {amount}.")

    def fold_action(self):
        # Implement fold logic
        print("Human folds.")
        # Handle fold in game logic

    def main_loop(self):
        running = True
        clock = pygame.time.Clock()
        while running:
            SCREEN.fill((0, 128, 0))  # Green background

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # Handle user input for betting, folding, etc.
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    # Check if buttons are clicked and handle actions

            # Update game state
            # For example, if it's AI's turn, let it decide and update the game

            # Draw hands
            self.draw_hand(self.game.players[0].hand, 50, SCREEN_HEIGHT - 130)  # Human
            if self.show_ai_hand:
                self.draw_hand(self.game.players[1].hand, 50, 70)  # AI
            else:
                # Draw AI's cards face down
                back_image = pygame.image.load(os.path.join('assets', 'cards', 'back.png'))
                for idx in range(2):
                    SCREEN.blit(back_image, (50 + idx * 75, 70))

            # Draw community cards
            self.draw_hand(self.game.community_cards, SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 30)

            # Draw pot size
            self.draw_text(f"Pot: {self.game.pot}", SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 + 50)

            # Draw player stacks
            for idx, player in enumerate(self.game.players):
                y_pos = SCREEN_HEIGHT - 200 if idx == 0 else 20
                self.draw_text(f"{player.name} Stack: {player.stack}", 50, y_pos)

            # Draw buttons for actions (Bet, Call, Fold)
            # Draw buttons
            self.draw_button("Bet", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 150, 100, 40, self.bet_action)
            self.draw_button("Call", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 100, 100, 40, self.call_action)
            self.draw_button("Fold", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 50, 100, 40, self.fold_action)


            pygame.display.flip()
            clock.tick(30)  # Limit to 30 FPS

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    gui = PokerGUI()
    gui.game.play_round()  # Start a game round
    gui.main_loop()
