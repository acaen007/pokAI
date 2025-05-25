from .hand import Hand
from ai.debug_utils import debug_print
class Player:
    def __init__(self, name, stack):
        self.name = name
        self.stack = stack
        self.hand = Hand()
        self.current_bet = 0

    def bet(self, amount):
        if amount > self.stack:
            debug_print("You do not have enough chips to bet that amount. Betting all in.")
        actual_bet = min(self.stack, amount)
        self.stack -= actual_bet
        return actual_bet

    def reset_bet(self):
        self.current_bet = 0

    def __repr__(self):
        return f"{self.name} with stack {self.stack}"
    

