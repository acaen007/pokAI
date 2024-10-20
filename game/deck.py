import numpy as np
from game.card import Card

class Deck:
    def __init__(self):
        self.cards = np.array([Card(suit, rank) for suit in Card.SUITS for rank in Card.RANKS])
        np.random.shuffle(self.cards)

    def deal(self):
        if len(self.cards) == 0:
            return None
        card = self.cards[-1]
        self.cards = np.delete(self.cards, -1)
        return card
