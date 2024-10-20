from game.deck import Deck
from ai.agent import PokerAI
import numpy as np


class PokerGame:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.pot = 0
        self.deck = None
        self.community_cards = []
        self.current_player_index = 0
        self.stage = 'not_started'  # Added to track the game stage
        self.bets_to_match = 0  # To track the current bet to match
    
    def start_new_round(self):
        print("Starting a new round.")
        self.reset()
        self.deck = Deck()
        self.deal_hole_cards()
        self.stage = 'pre_flop'
    
    # Update the reset method
    def reset(self):
        self.pot = 0
        self.deck = None
        self.community_cards = []
        for player in self.players:
            player.hand.reset()
            player.reset_bet()
        self.current_player_index = 0
        self.stage = 'not_started'
        self.bets_to_match = 0
    
    def next_stage(self):
        # Reset current bets and bets to match at the start of the new betting round
        for player in self.players:
            player.reset_bet()
        self.bets_to_match = 0

        if self.stage == 'not_started' or self.stage == 'complete':
            self.start_new_round()
            self.stage = 'pre_flop'
        elif self.stage == 'pre_flop':
            self.deal_community_cards(3)  # Flop
            self.stage = 'flop'
        elif self.stage == 'flop':
            self.deal_community_cards(1)  # Turn
            self.stage = 'turn'
        elif self.stage == 'turn':
            self.deal_community_cards(1)  # River
            self.stage = 'river'
        elif self.stage == 'river':
            self.stage = 'showdown'
        else:
            self.stage = 'complete'

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
        # Example game state representation
        state = np.zeros(10)  # Placeholder
        # Populate with actual game state data
        return state

    def handle_action(self, player, action, amount=0):
        if action == 'fold':
            # Handle folding
            print(f"{player.name} folds.")
            self.end_round(winner=self.get_other_player(player))
        elif action == 'call':
            call_amount = self.bets_to_match - player.current_bet
            bet_amount = player.bet(call_amount)
            self.pot += bet_amount
            print(f"{player.name} calls {call_amount}.")
        elif action == 'raise':
            raise_amount = amount
            total_bet = self.bets_to_match + raise_amount
            bet_amount = player.bet(total_bet - player.current_bet)
            self.pot += bet_amount
            self.bets_to_match = total_bet
            print(f"{player.name} raises to {total_bet}.")
        elif action == 'bet':
            bet_amount = amount
            bet_amount = player.bet(bet_amount)
            self.pot += bet_amount
            self.bets_to_match = bet_amount
            print(f"{player.name} bets {bet_amount}.")


    def get_other_player(self, player):
        return self.players[0] if self.players[1] == player else self.players[1]

    def end_round(self, winner=None):
        if winner:
            # Distribute pot to the winner
            winner.stack += self.pot
            print(f"{winner.name} wins the pot of {self.pot}.")
        else:
            # Handle showdown and determine winner
            winner = self.determine_winner()
            if winner:
                winner.stack += self.pot
                print(f"{winner.name} wins the pot of {self.pot}.")
            else:
                # Handle tie (split pot)
                split_amount = self.pot // 2
                for player in self.players:
                    player.stack += split_amount
                print(f"Pot is split between players.")
        self.reset()  # Prepare for the next round

    def determine_winner(self):
        # Implement hand evaluation logic to determine the winner
        # For now, randomly select a winner as a placeholder
        # Replace this with actual hand evaluation logic
        import random
        winner = random.choice(self.players)
        return winner
    
    def showdown(self):
        # Placeholder for hand evaluation logic
        print("Showdown between players")
        # Determine winner and distribute pot
