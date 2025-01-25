import pygame
import sys
import os
import requests
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from game.game import PokerGame
from game.player import Player
from game.card import Card
from game.hand import Hand
from game.deck import Deck
from game.hand_evaluator import (
    evaluate_hand,
    get_hand_rank_name,
    classify_hand,
    get_all_five_card_combinations
)

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Heads-Up Poker with Slumbot AI")

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
        image = pygame.image.load(image_path)
        image = pygame.transform.scale(image, (80, 100))  # Ensure consistent size
        card_images['back'] = image
    else:
        print(f"Warning: Image {back_image_path} not found.")
    return card_images


CARD_IMAGES = load_card_images()

# Slumbot Interface Class
class SlumbotInterface:
    def __init__(self, username=None, password=None):
        self.host = 'slumbot.com'
        self.token = None
        if username and password:
            self.token = self.login(username, password)
        else:
            self.token = None

    def login(self, username, password):
        data = {"username": username, "password": password}
        response = requests.post(f'https://{self.host}/api/login', json=data)
        if response.status_code != 200:
            print(f"Error logging in: {response.status_code}")
            sys.exit(-1)
        r = response.json()
        if 'error_msg' in r:
            print(f"Error: {r['error_msg']}")
            sys.exit(-1)
        token = r.get('token')
        if not token:
            print('Did not get token in response to /api/login')
            sys.exit(-1)
        return token

    def new_hand(self):
        data = {}
        if self.token:
            data['token'] = self.token
        response = requests.post(f'https://{self.host}/api/new_hand', headers={}, json=data)
        if response.status_code != 200:
            print(f"Error starting new hand: {response.status_code}")
            sys.exit(-1)
        r = response.json()
        if 'error_msg' in r:
            print(f"Error: {r['error_msg']}")
            sys.exit(-1)
        # Update token if provided
        self.token = r.get('token', self.token)
        return r

    def act(self, action):
        data = {'token': self.token, 'incr': action}
        response = requests.post(f'https://{self.host}/api/act', headers={}, json=data)
        if response.status_code != 200:
            print(f"Error acting: {response.status_code}")
            sys.exit(-1)
        r = response.json()
        if 'error_msg' in r:
            print(f"Error: {r['error_msg']}")
            sys.exit(-1)
        # Update token if provided
        self.token = r.get('token', self.token)
        return r

# Main GUI class
class PokerGUI:
    def __init__(self):
        self.game = PokerGame(
            Player('Human', 20000),
            Player('Slumbot', 20000)
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

        # Initialize Slumbot Interface
        self.slumbot = SlumbotInterface()  # Or provide username and password if needed
        self.token = self.slumbot.token
        self.human_last_action = None
        self.human_last_amount = 0

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

    def draw_dealer_button(self):
        dealer_pos = (100, SCREEN_HEIGHT - 230) if self.game.dealer == self.game.players[0] else (100, 220)
        pygame.draw.circle(SCREEN, (255, 215, 0), dealer_pos, 15)  # Gold-colored dealer button
        self.draw_text("D", dealer_pos[0] - 5, dealer_pos[1] - 8, color=(0, 0, 0))

    def draw_turn_arrow(self):
        arrow_pos = (250, SCREEN_HEIGHT - 100) if self.game.current_player_index == 0 else (250, 100)
        pygame.draw.polygon(SCREEN, (255, 255, 0), [(arrow_pos[0], arrow_pos[1]), (arrow_pos[0] - 10, arrow_pos[1] - 10), (arrow_pos[0], arrow_pos[1] - 20)])  # Yellow arrow

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

        # Draw the community cards
        for idx, card in enumerate(self.game.community_cards):
            x = start_x + idx * card_spacing
            is_highlighted = False
            if self.winning_hand:
                for highlight_card in self.winning_hand:
                    if card.rank == highlight_card.rank and card.suit == highlight_card.suit:
                        is_highlighted = True
                        break
            self.draw_card(card, x, y, highlight=is_highlighted)

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
                    pass
                else:
                    self.input_text += event.unicode

    def handle_user_input(self, x, y):
        # Check if buttons are clicked and handle actions
        for button_name, button_rect in self.button_rects.items():
            if button_rect.collidepoint(x, y):
                print(f"Button {button_name} clicked.")
                if button_name == 'Check':
                    self.human_last_action = 'check'
                    self.human_last_amount = 0
                    self.human_action_made = True
                elif button_name == 'Bet':
                    amount = int(self.input_text) if self.input_text.isdigit() else self.game.big_blind
                    self.human_last_action = 'bet'
                    self.human_last_amount = amount
                    self.human_action_made = True
                    self.input_text = ''
                elif button_name == 'Call':
                    self.human_last_action = 'call'
                    self.human_last_amount = 0
                    self.human_action_made = True
                elif button_name == 'Fold':
                    self.human_last_action = 'fold'
                    self.human_last_amount = 0
                    self.human_action_made = True
                elif button_name == 'Raise':
                    # Get the amount from the input text
                    try:
                        amount = int(self.input_text)
                        if amount > 0:
                            self.human_last_action = 'raise'
                            self.human_last_amount = amount
                            self.human_action_made = True
                            self.input_text = ''  # Clear the input
                        else:
                            print("Raise amount must be greater than 0.")
                    except ValueError:
                        print("Invalid raise amount.")
                elif button_name == 'Next Hand':
                    self.start_new_round()
                break  # Stop checking after the first matching button

    def map_human_action_to_slumbot(self, action, amount=0):
        if action == 'check':
            return 'k'
        elif action == 'call':
            return 'c'
        elif action in ['bet', 'raise']:
            return f"b{amount}"
        elif action == 'fold':
            return 'f'
        else:
            raise ValueError(f"Unknown action: {action}")

    def apply_slumbot_action(self, action_str):
        index = 0
        while index < len(action_str):
            c = action_str[index]
            index += 1
            if c == 'k':
                self.game.handle_action(self.game.players[1], 'check')
            elif c == 'c':
                self.game.handle_action(self.game.players[1], 'call')
            elif c == 'f':
                self.game.handle_action(self.game.players[1], 'fold')
            elif c == 'b':
                # Get the bet amount
                amount_str = ''
                while index < len(action_str) and action_str[index].isdigit():
                    amount_str += action_str[index]
                    index += 1
                amount = int(amount_str)
                self.game.handle_action(self.game.players[1], 'bet', amount=amount)
            elif c == '/':
                # Handle street separators
                pass
            else:
                raise ValueError(f"Unknown action character: {c}")

    def start_new_round(self):
        # Start a new hand with Slumbot
        response = self.slumbot.new_hand()
        # Update the token if necessary
        self.token = self.slumbot.token
        # Get the human player's hole cards
        hole_cards = response['hole_cards']
        # Convert card strings to Card objects
        human_hole_cards = [Card.from_string(card_str) for card_str in hole_cards]
        self.game.players[0].hand.cards = human_hole_cards
        # The AI player's hole cards are unknown at this point
        self.game.players[1].hand.cards = []  # Empty list or placeholder
        # Reset game states
        self.game.pot = 0
        self.game.community_cards = []
        self.game.stage = 'pre_flop'
        self.human_action_made = False
        self.show_ai_hand = False
        self.winner = None
        self.winning_hand = []
        self.hand_rank = ''
        # Clear the button_rects dictionary
        self.button_rects = {}
        # Apply any initial action from Slumbot
        initial_action = response.get('action', '')
        if initial_action:
            self.apply_slumbot_action(initial_action)
        # Update community cards if any
        board_cards = response.get('board', [])
        self.game.community_cards = [Card.from_string(card_str) for card_str in board_cards]

    def update_game_state(self):
        if self.game.current_player_index == 0 and self.human_action_made:
            # Map human action to Slumbot action string
            human_action_str = self.map_human_action_to_slumbot(
                self.human_last_action,
                amount=self.human_last_amount
            )
            # Send human action to Slumbot
            response = self.slumbot.act(human_action_str)
            self.token = self.slumbot.token
            # Apply human action to game state
            self.game.handle_action(self.game.players[0], self.human_last_action, amount=self.human_last_amount)
            # Apply Slumbot's action
            slumbot_action = response.get('action', '')
            if slumbot_action:
                self.apply_slumbot_action(slumbot_action)
            # Update community cards
            board_cards = response.get('board', [])
            self.game.community_cards = [Card.from_string(card_str) for card_str in board_cards]
            # Check if the hand is over
            if 'winnings' in response:
                # Handle end of hand
                self.game.stage = 'complete'
                self.show_ai_hand = True
                # Update AI's hole cards if revealed
                ai_hole_cards = response.get('opponent_hole_cards', [])
                if ai_hole_cards:
                    self.game.players[1].hand.cards = [Card.from_string(card_str) for card_str in ai_hole_cards]
                # Determine winner
                winnings = response.get('winnings')
                if winnings > 0:
                    self.winner = self.game.players[0]
                    self.hand_rank = "Win"
                elif winnings < 0:
                    self.winner = self.game.players[1]
                    self.hand_rank = "Lose"
                else:
                    self.winner = None  # Tie
                    self.hand_rank = "Tie"
                # Evaluate winning hand if showdown
                if self.winner:
                    self.winning_hand = self.game.get_best_five_card_hand(
                        self.winner.hand.cards,
                        self.game.community_cards
                    )
                    rank, _ = evaluate_hand(self.winner.hand.cards, self.game.community_cards)
                    self.hand_rank = get_hand_rank_name(rank)
            self.human_action_made = False

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
        # Draw dealer button and turn arrow
        self.draw_dealer_button()
        self.draw_turn_arrow()

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
            self.draw_text("Bet/Raise Amount:", self.input_rect.x - 40, self.input_rect.y - 75)

            # Draw action buttons
            if self.game.current_player_index == 0 and not self.human_action_made:
                check_button_rect = self.draw_button("Check", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 200, 100, 40)
                self.button_rects['Check'] = check_button_rect
                bet_button_rect = self.draw_button("Bet", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 150, 100, 40)
                self.button_rects['Bet'] = bet_button_rect
                call_button_rect = self.draw_button("Call", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 100, 100, 40)
                self.button_rects['Call'] = call_button_rect
                fold_button_rect = self.draw_button("Fold", SCREEN_WIDTH - 150, SCREEN_HEIGHT - 50, 100, 40)
                self.button_rects['Fold'] = fold_button_rect
                raise_button_rect = self.draw_button("Raise", SCREEN_WIDTH - 250, SCREEN_HEIGHT - 100, 80, 40)
                self.button_rects['Raise'] = raise_button_rect

        if not self.running:
            self.draw_text("Game Over", SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2, color=(255, 0, 0))

    def main_loop(self):
        # Start a new round
        self.start_new_round()

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
