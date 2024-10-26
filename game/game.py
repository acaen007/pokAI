# game.py

from game.deck import Deck
from game.hand import Hand
from game.player import Player
from game.hand_evaluator import evaluate_hand

class PokerGame:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.pot = 0
        self.deck = None
        self.community_cards = []
        self.current_player_index = 0
        self.stage = 'not_started'
        self.bets_to_match = 0
        self.previous_actions = []
        self.small_blind = 10
        self.big_blind = 20
        self.player_all_in = None  # Track if a player is all-in
        self.players_who_acted = set()  # Initialize players_who_acted
        self.actions_in_round = 0  # Initialize action count for betting rounds

    def start_new_round(self):
        print("\nStarting a new round.")
        print("player_stacks:", [player.stack for player in self.players])
        self.reset_between_rounds()
        self.deck = Deck()
        self.deal_hole_cards()
        self.stage = 'pre_flop'
        self.post_blinds()
        print("Betting Round: Pre-flop")

    def post_blinds(self):
        small_blind_player = self.players[0]
        big_blind_player = self.players[1]

        # Small Blind
        small_blind_amount = min(self.small_blind, small_blind_player.stack)
        small_blind_player.bet(small_blind_amount)
        small_blind_player.current_bet = small_blind_amount
        self.pot += small_blind_amount
        print(f"{small_blind_player.name} posts small blind of {small_blind_amount}.")

        # Big Blind
        big_blind_amount = min(self.big_blind, big_blind_player.stack)
        big_blind_player.bet(big_blind_amount)
        big_blind_player.current_bet = big_blind_amount
        self.pot += big_blind_amount
        print(f"{big_blind_player.name} posts big blind of {big_blind_amount}.")

        # Set bets to match
        self.bets_to_match = max(small_blind_player.current_bet, big_blind_player.current_bet)

        # Check for all-in players
        if small_blind_player.stack == 0:
            self.player_all_in = small_blind_player
            print(f"{small_blind_player.name} is all-in.")
        if big_blind_player.stack == 0:
            self.player_all_in = big_blind_player
            print(f"{big_blind_player.name} is all-in.")

        # Action starts with the player after the big blind
        self.current_player_index = (self.players.index(big_blind_player) + 1) % len(self.players)

    def reset_between_rounds(self):
        # Reset attributes between rounds but keep players' stacks
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
        self.player_all_in = None
        self.players_who_acted = set()
        self.actions_in_round = 0

    def reset_actions_in_round(self):
        self.actions_in_round = 0
        self.players_who_acted = set()

    def next_stage(self):
        # Reset current bets and bets to match at the start of the new betting round
        for player in self.players:
            player.current_bet = 0
        self.bets_to_match = 0
        self.reset_actions_in_round()  # Reset at the start of each betting round

        if self.stage == 'pre_flop':
            self.deal_community_cards(3)  # Flop
            self.stage = 'flop'
            print("Betting Round: Flop")
            print("Community Cards:", self.format_cards(self.community_cards))
        elif self.stage == 'flop':
            self.deal_community_cards(1)  # Turn
            self.stage = 'turn'
            print("Betting Round: Turn")
            print("Community Cards:", self.format_cards(self.community_cards))
        elif self.stage == 'turn':
            self.deal_community_cards(1)  # River
            self.stage = 'river'
            print("Betting Round: River")
            print("Community Cards:", self.format_cards(self.community_cards))
        elif self.stage == 'river':
            self.stage = 'showdown'
            print("Proceeding to Showdown")
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
            'action_history': self.previous_actions,
            'bets_to_match': self.bets_to_match,
            'player_stack': player.stack,
            'current_bet': player.current_bet,
            'betting_allowed': self.is_betting_allowed()
        }
        return state

    def handle_action(self, player, action, amount=0, action_index=None):
        player_index = self.players.index(player)
        bets_to_match = self.bets_to_match
        current_bet = player.current_bet

        # Validate action legality
        if action == 'check' and bets_to_match > current_bet:
            # Illegal action: cannot check when there's an outstanding bet
            print(f"{player.name} cannot check when there's an outstanding bet.")
            # For simplicity, we'll treat this as a fold here
            action = 'fold'

        if action == 'fold':
            print(f"{player.name} folds.")
            winner = self.get_other_player(player)
            self.end_round(winner=winner)
            return  # Round ends when a player folds
        elif action == 'check':
            print(f"{player.name} checks.")
        elif action == 'call':
            call_amount = self.bets_to_match - player.current_bet
            bet_amount = min(call_amount, player.stack)
            actual_bet = player.bet(bet_amount)
            self.pot += actual_bet
            player.current_bet += actual_bet
            print(f"{player.name} calls {actual_bet}.")
            if player.stack == 0:
                self.player_all_in = player
                print(f"{player.name} is all-in.")
        elif action in ['bet', 'raise']:
            total_bet = amount
            bet_amount = total_bet - player.current_bet
            # Ensure bet_amount does not exceed player's stack
            bet_amount = min(bet_amount, player.stack)
            if bet_amount <= 0:
                print(f"{player.name} attempts to {action} with invalid total bet amount {total_bet}.")
                bet_amount = min(self.big_blind, player.stack)
                total_bet = player.current_bet + bet_amount
            else:
                total_bet = player.current_bet + bet_amount
            actual_bet = player.bet(bet_amount)
            self.pot += actual_bet
            self.bets_to_match = max(self.bets_to_match, total_bet)
            player.current_bet = total_bet
            print(f"{player.name} {action}s to {total_bet}.")
            if player.stack == 0:
                self.player_all_in = player
                print(f"{player.name} is all-in.")
        else:
            print(f"Unknown action: {action}")

        if action_index is None:
            action_index = 0  # Default action index if not provided
        action_info = {'player_index': player_index, 'action_index': action_index}
        self.previous_actions.append(action_info)

        # Record that the player has acted
        self.players_who_acted.add(player)

    def get_other_player(self, player):
        return self.players[0] if self.players[1] == player else self.players[1]

    def end_round(self, winner=None):
        if winner:
            # Distribute pot to the winner
            winner.stack += self.pot
            print(f"{winner.name} wins the pot of {self.pot}.")
            # Show winner's hand
            winner_hand = winner.hand.cards + self.community_cards
            print(f"{winner.name}'s hand: {self.format_cards(winner_hand)}")
        else:
            # Handle showdown and determine winner
            winner = self.determine_winner()
            if winner:
                winner.stack += self.pot
                print(f"{winner.name} wins the pot of {self.pot}.")
                # Show winner's hand
                winner_hand = winner.hand.cards + self.community_cards
                print(f"{winner.name}'s hand: {self.format_cards(winner_hand)}")
            else:
                # Handle tie (split pot)
                split_amount = self.pot // 2
                for player in self.players:
                    player.stack += split_amount
                    player_hand = player.hand.cards + self.community_cards
                    print(f"{player.name}'s hand: {self.format_cards(player_hand)}")
                print(f"Pot is split between players.")

        # Print each player's stack
        for player in self.players:
            print(f"{player.name} stack: {player.stack}")

        # Reset the pot to zero
        self.pot = 0  # Ensure pot is reset after distributing winnings

        # Do not reset the deck here
        # Prepare for the next round by resetting attributes but keep the deck until the round actually starts
        self.reset_between_rounds()
        
        # Set stage to 'complete' to indicate that the round is over
        self.stage = 'complete'


    def format_cards(self, cards):
        return ', '.join([f"{card.rank} of {card.suit}" for card in cards])

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
        # Get the current bets of all players
        bets = [player.current_bet for player in self.players]
        max_bet = max(bets)
        # Players who can act (have chips left)
        active_players = [player for player in self.players if player.stack > 0]
        for player in active_players:
            if player.current_bet < max_bet:
                return False
        return True

    def all_players_acted(self):
        # All players have acted if each player has made at least one action in this betting round
        return self.actions_in_round >= len(self.players)

    def is_betting_allowed(self):
        # Betting is allowed if not all players are all-in
        active_players = [player for player in self.players if player.stack > 0]
        return len(active_players) > 1

    def should_move_to_next_stage(self):
        """
        Determines whether the betting round should end and move to the next stage.
        The betting round ends when:
        - All players have acted at least once in the current betting round, and
          - All players have checked (no bets were made), or
          - A bet has been made and then called or all-in has been called.
        """
        # If all players have acted at least once
        if self.actions_in_round >= len(self.players):
            # If all players have checked
            if self.bets_to_match == 0 and all(player.current_bet == 0 for player in self.players):
                print("All players have checked. Moving to next stage.")
                return True
            # If bets are matched (including all-ins)
            elif self.players_matched_bets():
                print("Bets are matched. Moving to next stage.")
                return True
        return False
