from game.deck import Deck


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
        # Simplified betting logic for demonstration
        for player in self.players:
            bet_amount = 10  # Placeholder for actual betting logic
            self.pot += player.bet(bet_amount)
            print(f"{player.name} bets {bet_amount}")

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
