from game.deck import Deck
from ai.agent import PokerAI
import numpy as np
from game.hand_evaluator import evaluate_hand



class PokerGame:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.pot = 0
        self.deck = None
        self.community_cards = []
        self.current_player_index = 0
        self.stage = 'not_started'  # Added to track the game stage
        self.bets_to_match = 0  # To track the current bet to match
        self.previous_actions = []  # Keep track of actions

    
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
            player.current_bet = 0
        self.current_player_index = 0
        self.stage = 'not_started'
        self.bets_to_match = 0
        self.previous_actions = []

    
    def next_stage(self):
        # Reset current bets and bets to match at the start of the new betting round
        for player in self.players:
            player.current_bet = 0
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
        state = {
            'player_hand': player.hand.cards,
            'community_cards': self.community_cards,
            'pot': self.pot,
            'previous_actions': self.previous_actions,  # You need to keep track of actions
            'bets_to_match': self.bets_to_match
        }
        return state

    def handle_action(self, player, action, amount=0):
        if action == 'fold':
        # Handle folding
            print(f"{player.name} folds.")
            winner = self.get_other_player(player)
            self.end_round(winner=winner)
        elif action == 'call':
            call_amount = self.bets_to_match - player.current_bet
            bet_amount = player.bet(call_amount)
            self.pot += bet_amount
            print(f"{player.name} calls {call_amount}.")
            player.current_bet += bet_amount
        elif action == 'raise' or action == 'bet':
            total_bet = amount
            bet_amount = player.bet(total_bet - player.current_bet)
            self.pot += bet_amount
            self.bets_to_match = total_bet
            print(f"{player.name} {action}s to {total_bet}.")
            player.current_bet += bet_amount
        elif action == 'check':
            print(f"{player.name} checks.")
        action_code = self.action_to_code(action)
        self.previous_actions.append(action_code)

    def action_to_code(self, action):
        action_mapping = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4}
        return action_mapping.get(action, -1)



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
        player_hands = []
        for player in self.players:
            hand_rank, high_cards = evaluate_hand(player.hand.cards, self.community_cards)
            player_hands.append((player, hand_rank, high_cards))

        # Compare hand ranks
        player_hands.sort(key=lambda x: (x[1], x[2]), reverse=True)
        # Check for tie
        if player_hands[0][1:] == player_hands[1][1:]:
            print("It's a tie!")
            return None  # Indicate a tie
        else:
            winner = player_hands[0][0]
            print(f"{winner.name} wins with hand rank {player_hands[0][1]} and high cards {player_hands[0][2]}")
            return winner
        
    def players_matched_bets(self):
        # Get the current bets of all active players
        bets = [player.current_bet for player in self.players]
        # Check if all bets are equal
        return all(bet == bets[0] for bet in bets)
    
    # def showdown(self):
    #     # Placeholder for hand evaluation logic
    #     print("Showdown between players")
    #     # Determine winner and distribute pot
