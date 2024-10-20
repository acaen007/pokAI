from game.hand import Hand
class Player:
    def __init__(self, name, stack):
        self.name = name
        self.stack = stack
        self.hand = Hand()
        self.current_bet = 0

    def bet(self, amount):
        amount = min(amount, self.stack)
        self.stack -= amount
        self.current_bet += amount
        return amount

    def reset_bet(self):
        self.current_bet = 0

    def __repr__(self):
        return f"{self.name} with stack {self.stack}"
    

