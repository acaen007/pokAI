from game.hand import Hand
class Player:
    def __init__(self, name, stack):
        self.name = name
        self.stack = stack
        self.hand = Hand()
        self.current_bet = 0

    def bet(self, amount):
        if amount <= 0:
            raise ValueError("Bet amount must be greater than zero.")
        actual_bet = min(self.stack, amount)
        self.stack -= actual_bet
        return actual_bet

    def reset_bet(self):
        self.current_bet = 0

    def __repr__(self):
        return f"{self.name} with stack {self.stack}"
    

