import pygame
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from ai.agent import PokerAI
from game.game import PokerGame
from game.player import Player
from game.card import Card

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Heads-Up Poker AI")

# Load card images
def load_card_images():
    card_images = {}
    base_path = os.path.dirname(__file__)
    assets_path = os.path.join(base_path, '..', 'assets', 'cards')
    suits = ['hearts', 'diamonds', 'clubs', 'spades']
    ranks = [str(n) for n in range(2, 11)] + ['J', 'Q', 'K', 'A']
    for suit in suits:
        for rank in ranks:
            card_name = f"{suit.lower()}_{rank}"
            image_path = os.path.join(assets_path, f"{card_name}.png")
            if os.path.exists(image_path):
                image = pygame.image.load(image_path)
                # Resize the image
                image = pygame.transform.scale(image, (80, 100))  # Adjust the size as needed
                card_images[card_name] = image
            else:
                print(f"Warning: Image {image_path} not found.")
    # Load the back of card image
    back_image_path = os.path.join(assets_path, 'back.png')
    if os.path.exists(back_image_path):
        image = pygame.image.load(back_image_path)
        image = pygame.transform.scale(image, (80, 100))  # Ensure consistent size
        card_images['back'] = image
    else:
        print(f"Warning: Image {back_image_path} not found.")
    return card_images


CARD_IMAGES = load_card_images()

# Main GUI class
class PokerGUI:
    def __init__(self):
        self.game = PokerGame(
            Player('Human', 1000),
            PokerAI('AI', 1000)
        )
        self.font = pygame.font.SysFont(None, 24)
        self.show_ai_hand = False
        self.running = True
        self.clock = pygame.time.Clock()
        self.human_action_made = False  # To track if human has made an action
        self.input_text = ''
        self.active_input = False
        self.input_rect = pygame.Rect(SCREEN_WIDTH - 250, SCREEN_HEIGHT - 50, 80, 30)
        self.button_rects = {}  # To store button rectangles for event handling
        self.winner = None
        self.winning_hand = []
        self.hand_rank = ''
    
    def draw_text(self, text, x, y, color=(255, 255, 255)):
        img = self.font.render(text, True, color)
        SCREEN.blit(img, (x, y))
    
    def draw_button(self, text, x, y, width, height):
        mouse = pygame.mouse.get_pos()
        button_rect = pygame.Rect(x, y, width, height)

        if button_rect.collidepoint(mouse):
            color = (170, 170, 170)  # Hover color
        else:
            color = (100, 100, 100)  # Normal color

        pygame.draw.rect(SCREEN, color, button_rect)
        self.draw_text(text, x + 10, y + 10)

        return button_rect

    def draw_card(self, card, x, y, highlight=False):
        card_name = f"{card.suit}_{card.rank}"
        image = CARD_IMAGES.get(card_name)
        if image:
            if highlight:
                # Draw a yellow rectangle behind the card to highlight it
                highlight_rect = pygame.Rect(x - 5, y - 5, image.get_width() + 10, image.get_height() + 10)
                pygame.draw.rect(SCREEN, (255, 255, 0), highlight_rect)
            SCREEN.blit(image, (x, y))
        else:
            # Draw a placeholder rectangle if image not found
            pygame.draw.rect(SCREEN, (255, 0, 0), (x, y, 71, 96))  # Standard card size

    def draw_hand(self, cards, x, y, highlight_cards=None):
        for idx, card in enumerate(cards):
            is_highlighted = False
            if highlight_cards:
                for highlight_card in highlight_cards:
                    if card.rank == highlight_card.rank and card.suit == highlight_card.suit:
                        is_highlighted = True
                        break
            self.draw_card(card, x + idx * 100, y, highlight=is_highlighted)
            
    def draw_community_cards(self):
        # Positions for the community cards
        start_x = SCREEN_WIDTH // 2 - 270
        y = SCREEN_HEIGHT // 2 - 30
        card_spacing = 100  # Adjust spacing based on card size

        # Total number of community cards
        total_community_cards = 5

        # Determine the number of cards to reveal based on the game stage
        if self.game.stage == 'pre_flop':
            revealed_cards = 0
        elif self.game.stage == 'flop':
            revealed_cards = 3
        elif self.game.stage == 'turn':
            revealed_cards = 4
        elif self.game.stage in ['river', 'showdown', 'complete']:
            revealed_cards = 5
        else:
            revealed_cards = 0

        # Draw the community cards
        for i in range(total_community_cards):
            x = start_x + i * card_spacing
            if i < revealed_cards and i < len(self.game.community_cards):
                # Draw the revealed card
                card = self.game.community_cards[i]
                is_highlighted = False
                if self.winning_hand:
                    for highlight_card in self.winning_hand:
                        if card.rank == highlight_card.rank and card.suit == highlight_card.suit:
                            is_highlighted = True
                            break
                self.draw_card(card, x, y, highlight=is_highlighted)
            else:
                # Draw the back of the card
                back_image = CARD_IMAGES.get('back')
                back_image = pygame.transform.scale(back_image, (80, 100))  # Ensure consistent size
                if back_image:
                    SCREEN.blit(back_image, (x, y))
                else:
                    pygame.draw.rect(SCREEN, (255, 0, 0), (x, y, 50, 70))

    def bet_action(self):
        # Implement betting logic
        amount = self.game.small_blind  # Placeholder amount
        self.game.players[0].bet(amount)
        self.game.pot += amount
        print(f"Human bets {amount}.")

    def call_action(self):
        # Implement call logic
        amount = self.game.current_bet - self.game.players[0].current_bet
        self.game.players[0].bet(amount)
        self.game.pot += amount
        print(f"Human calls {amount}.")

    def fold_action(self):
        # Implement fold logic
        print("Human folds.")
        self.game.handle_action(self.game.players[0], 'fold')
        self.human_action_made = True

    def getHumanAction(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                self.handle_user_input(x, y)
                    # Check if the input box is clicked
                if self.input_rect.collidepoint(event.pos):
                    self.active_input = True
                else:
                    self.active_input = False
            elif event.type == pygame.KEYDOWN and self.active_input:
                if event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                elif event.key == pygame.K_RETURN:
                        # You can handle Enter key if needed
                    pass
                else:
                    self.input_text += event.unicode

    def handle_user_input(self, x, y):
        # Check if buttons are clicked and handle actions
        for button_name, button_rect in self.button_rects.items():
            if button_rect.collidepoint(x, y):
                print(f"Button {button_name} clicked.")
                if button_name == 'Check':
                    self.human_action('check')
                elif button_name == 'Bet':
                    self.human_action('bet', amount=10)
                elif button_name == 'Call':
                    self.human_action('call')
                elif button_name == 'Fold':
                    self.human_action('fold')
                elif button_name == 'Raise':
                    # Get the amount from the input text
                    try:
                        amount = int(self.input_text)
                        if amount > 0:
                            self.human_action('raise', amount=amount)
                            self.input_text = ''  # Clear the input
                        else:
                            print("Raise amount must be greater than 0.")
                    except ValueError:
                        print("Invalid raise amount.")
                elif button_name == 'Next Hand':
                    # Reset game state for the next hand

                    #if either player has run out of chips, reset the stacks
                    if self.game.players[0].stack <= 0 or self.game.players[1].stack <= 0:
                        self.game.players[0].stack = 1000
                        self.game.players[1].stack = 1000
                    self.game.start_new_round()
                    self.human_action_made = False
                    self.show_ai_hand = False
                    self.game.stage = 'pre_flop'
                    self.winner = None
                    self.winning_hand = []
                    self.hand_rank = ''
                    # Clear the button_rects dictionary
                    self.button_rects = {}
                break  # Stop checking after the first matching button

    def human_action(self, action, amount=0):
        if not self.human_action_made:
            self.game.handle_action(self.game.players[0], action, amount=amount)
            self.human_action_made = True  # Prevent multiple actions in one turn

    def players_matched_bets(self):
        bets = [player.current_bet for player in self.game.players]
        return bets[0] == bets[1]

    def update_game_state(self):
        if self.game.stage in ['pre_flop', 'flop', 'turn', 'river']:

            # Check if both players have matched bets
            if self.players_matched_bets() and len(self.game.players_who_acted)== 2:
                # Move to next stage
                self.game.next_stage()
                self.human_action_made = False

            if self.game.current_player_index == 1:
                # AI's turn
                ai_player = self.game.players[1]
                game_state = self.game.get_game_state(ai_player)
                ai_action = ai_player.decide_action(game_state)
                action_str = self.map_action(ai_action)
                self.game.handle_action(ai_player, action_str)
                self.human_action_made = False

        elif self.game.stage == 'showdown':
            # Showdown logic
            self.show_ai_hand = True
            self.winner, self.winning_hand, self.hand_rank = self.game.end_round()
            self.human_action_made = False
            self.game.stage = 'complete'
        elif self.game.stage == 'complete':
            # Waiting for 'Next Hand' button press
            pass

    def draw_game_elements(self):
        # Clear existing buttons
        self.button_rects = {}

        # Draw human player's hand
        self.draw_text("Your Hand:", 50, SCREEN_HEIGHT - 150)
        human_highlights = [card for card in self.game.players[0].hand.cards if card in self.winning_hand]
        self.draw_hand(self.game.players[0].hand.cards, 50, SCREEN_HEIGHT - 130, highlight_cards=human_highlights)

        # Draw AI player's hand
        self.draw_text("AI's Hand:", 50, 50)
        if self.show_ai_hand:
            ai_highlights = [card for card in self.game.players[1].hand.cards if card in self.winning_hand]
            self.draw_hand(self.game.players[1].hand.cards, 50, 70, highlight_cards=ai_highlights)
        else:
            # Draw face-down cards
            back_image = CARD_IMAGES.get('back')
            back_image = pygame.transform.scale(back_image, (80, 100))  # Ensure consistent size

            for idx in range(2):
                SCREEN.blit(back_image, (50 + idx * 100, 70))

        # Draw community cards
        self.draw_text("Community Cards:", SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 50)
        self.draw_community_cards()

        # Draw pot size
        self.draw_text(f"Pot: {self.game.pot}", SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 + 100)

        # Draw player stacks
        for idx, player in enumerate(self.game.players):
            y_pos = SCREEN_HEIGHT - 200 if idx == 0 else 20
            self.draw_text(f"{player.name} Stack: {player.stack}", 50, y_pos)

        if self.game.stage == 'complete':
            # Display winner and hand rank
            if self.winner:
                winner_name = self.winner.name
                self.draw_text(f"{winner_name} wins with {self.hand_rank}", SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 150)
            else:
                self.draw_text("It's a tie!", SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 + 150)

            # Draw the "Next Hand" button
            next_hand_button_rect = self.draw_button("Next Hand", SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 + 200, 100, 40)
            self.button_rects['Next Hand'] = next_hand_button_rect
        else:
            # Draw the input box
            pygame.draw.rect(SCREEN, (255, 255, 255), self.input_rect)
            # Draw the input text
            text_surface = self.font.render(self.input_text, True, (0, 0, 0))
            SCREEN.blit(text_surface, (self.input_rect.x + 5, self.input_rect.y + 5))
            
            # Draw a label for the input box
            self.draw_text("Raise Amount:", self.input_rect.x-40, self.input_rect.y - 75)

            # Draw action buttons
            if (self.game.current_player_index == 0 and self.game.bets_to_match == self.game.players[0].current_bet and self.game.stage != 'pre_flop') or (self.game.current_player_index == 0 and self.game.stage == 'pre_flop' and self.game.bets_to_match == self.game.players[0].current_bet): 
                check_button_rect = self.draw_button("Check", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 200, 100, 40)
                self.button_rects['Check'] = check_button_rect
            bet_button_rect = self.draw_button("Bet", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 150, 100, 40)
            call_button_rect = self.draw_button("Call", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 100, 100, 40)
            fold_button_rect = self.draw_button("Fold", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 50, 100, 40)
            raise_button_rect = self.draw_button("Raise", SCREEN_WIDTH - 250, SCREEN_HEIGHT - 100, 80, 40)

            # Store button rectangles for event handling
            self.button_rects['Bet'] = bet_button_rect
            self.button_rects['Call'] = call_button_rect
            self.button_rects['Fold'] = fold_button_rect
            self.button_rects['Raise'] = raise_button_rect

        if not self.running:
            self.draw_text("Game Over", SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2, color=(255, 0, 0))

    # Helper methods
    def is_button_clicked(self, button_name, x, y):
        # Define button positions and sizes
        buttons = {
            'Check': pygame.Rect(SCREEN_WIDTH - 150, SCREEN_HEIGHT - 200, 100, 40),
            'Bet': pygame.Rect(SCREEN_WIDTH - 150, SCREEN_HEIGHT - 150, 100, 40),
            'Call': pygame.Rect(SCREEN_WIDTH - 150, SCREEN_HEIGHT - 100, 100, 40),
            'Fold': pygame.Rect(SCREEN_WIDTH - 150, SCREEN_HEIGHT - 50, 100, 40),
        }
        button_rect = buttons.get(button_name)
        if button_rect and button_rect.collidepoint(x, y):
            return True
        return False

    def map_action(self, action_index):
        # Map AI action index to action string
        action_mapping = {0: 'fold', 1: 'call', 2: 'raise'}
        return action_mapping.get(action_index, 'call')
    
    def main_loop(self):
        # Start a new round
        self.game.start_new_round()
        
        while self.running:
            SCREEN.fill((0, 128, 0))  # Green background
                        # Draw all game elements
            self.draw_game_elements()

            self.getHumanAction()

            # Update game state based on stages
            self.update_game_state()

            pygame.display.flip()
            self.clock.tick(60)  # Limit to 60 FPS

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    gui = PokerGUI()
    gui.main_loop()
