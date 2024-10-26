# game.py
from game.deck import Deck
from ai.agent import PokerAI
import numpy as np
from game.hand_evaluator import evaluate_hand, get_hand_rank_name, classify_hand, get_all_five_card_combinations

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
        self.winner_declared = False  # To track if winner is declared early (e.g., on fold)
        self.winner = None  # To store the winner when declared early

    def start_new_round(self):
        print("Starting a new round.")
        self.reset()
        self.deck = Deck()
        self.deal_hole_cards()
        self.stage = 'pre_flop'

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
        self.winner_declared = False
        self.winner = None

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

    def get_game_state(self, player):
        state = {
            'player_hand': player.hand.cards,
            'community_cards': self.community_cards,
            'pot': self.pot,
            'previous_actions': self.previous_actions,
            'bets_to_match': self.bets_to_match
        }
        return state

    def handle_action(self, player, action, amount=0):
        if action == 'fold':
            # Handle folding
            print(f"{player.name} folds.")
            winner = self.get_other_player(player)
            self.winner_declared = True  # Indicate that the winner has been declared
            self.winner = winner  # Store the winner
            self.stage = 'showdown'  # Move directly to showdown
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

    def end_round(self):
        if self.winner_declared:
            # Winner was declared due to fold
            winner = self.winner
            winning_hand = []
            hand_rank = 'Opponent Folded'
        else:
            # Evaluate both players' hands
            player1_rank, player1_high_cards = evaluate_hand(self.players[0].hand.cards, self.community_cards)
            player2_rank, player2_high_cards = evaluate_hand(self.players[1].hand.cards, self.community_cards)

            # Compare hands to determine the winner
            if player1_rank > player2_rank:
                winner = self.players[0]
                winning_hand = self.get_best_five_card_hand(self.players[0].hand.cards, self.community_cards)
                hand_rank = get_hand_rank_name(player1_rank)
            elif player2_rank > player1_rank:
                winner = self.players[1]
                winning_hand = self.get_best_five_card_hand(self.players[1].hand.cards, self.community_cards)
                hand_rank = get_hand_rank_name(player2_rank)
            else:
                # If ranks are equal, compare high cards
                if player1_high_cards > player2_high_cards:
                    winner = self.players[0]
                    winning_hand = self.get_best_five_card_hand(self.players[0].hand.cards, self.community_cards)
                    hand_rank = get_hand_rank_name(player1_rank)
                elif player2_high_cards > player1_high_cards:
                    winner = self.players[1]
                    winning_hand = self.get_best_five_card_hand(self.players[1].hand.cards, self.community_cards)
                    hand_rank = get_hand_rank_name(player2_rank)
                else:
                    winner = None  # Tie
                    winning_hand = self.get_best_five_card_hand(self.players[0].hand.cards, self.community_cards)
                    hand_rank = get_hand_rank_name(player1_rank)

        # Update stacks and distribute pot
        if winner:
            winner.stack += self.pot
        else:
            # Split the pot in case of a tie
            split_amount = self.pot // 2
            self.players[0].stack += split_amount
            self.players[1].stack += split_amount

        # Reset pot
        self.pot = 0

        # Return the winner, the winning hand, and the hand rank
        return winner, winning_hand, hand_rank

    def get_best_five_card_hand(self, hand_cards, community_cards):
        # Combine all cards
        all_cards = hand_cards + community_cards
        all_combinations = get_all_five_card_combinations(all_cards)
        best_rank = -1
        best_high_cards = []
        best_hand = None
        for combo in all_combinations:
            rank, high_cards = classify_hand(combo)
            if rank > best_rank or (rank == best_rank and high_cards > best_high_cards):
                best_rank = rank
                best_high_cards = high_cards
                best_hand = combo
        return best_hand

    def players_matched_bets(self):
        # Get the current bets of all active players
        bets = [player.current_bet for player in self.players]
        # Check if all bets are equal
        return all(bet == bets[0] for bet in bets)
