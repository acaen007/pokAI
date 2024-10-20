from game.deck import Deck
from ai.agent import PokerAI
import numpy as np


class PokerGame:
    def __init__(self, player1, player2, starting_stack=1000):
        self.players = [player1, player2]
        self.pot = 0
        self.deck = Deck()
        self.community_cards = []
        self.current_player = 0  # Index of the player whose turn it is

    def reset(self):
        self.pot = 0
        self.deck = Deck()
        self.community_cards = []
        for player in self.players:
            player.hand.reset()
            player.reset_bet()

    def deal_hole_cards(self):
        for _ in range(2):
            for player in self.players:
                player.hand.add_card(self.deck.deal())

    def deal_community_cards(self, number):
        for _ in range(number):
            self.community_cards.append(self.deck.deal())

    def betting_round(self):
        # Implement betting logic
        for player in self.players:
            if isinstance(player, PokerAI):
                game_state = self.get_game_state(player)
                action = player.decide_action(game_state)
                # Map action index to actual action
                if action == 0:
                    print(f"{player.name} folds.")
                    # Handle fold
                elif action == 1:
                    amount = 10  # Placeholder for call amount
                    self.pot += player.bet(amount)
                    print(f"{player.name} calls {amount}.")
                elif action == 2:
                    amount = 20  # Placeholder for raise amount
                    self.pot += player.bet(amount)
                    print(f"{player.name} raises to {amount}.")
            else:
                # Human player logic or simplified AI
                amount = 10  # Placeholder
                self.pot += player.bet(amount)
                print(f"{player.name} bets {amount}.")
    
    def get_game_state(self, player):
        # Create a representation of the game state for the AI
        # For simplicity, we'll use a NumPy array of zeros
        game_state = np.zeros(114)  # Placeholder size
        # Populate game_state with relevant information
        # Example: Encode own cards, community cards, pot size, etc.
        return game_state

    def play_round(self):
        self.reset()
        self.deal_hole_cards()
        print("Hole Cards Dealt")
        self.betting_round()

        # Flop
        self.deal_community_cards(3)
        print("Flop:", self.community_cards)
        self.betting_round()

        # Turn
        self.deal_community_cards(1)
        print("Turn:", self.community_cards[-1])
        self.betting_round()

        # River
        self.deal_community_cards(1)
        print("River:", self.community_cards[-1])
        self.betting_round()

        # Showdown
        self.showdown()

    def showdown(self):
        # Placeholder for hand evaluation logic
        print("Showdown between players")
        # Determine winner and distribute pot
