# AI/card_representation.py

import numpy as np

class CardRepresentation:
    """
    Incrementally build a 6 x 4 x 13 card tensor:
      - Channel 0: hole cards
      - Channel 1: flop
      - Channel 2: turn
      - Channel 3: river
      - Channel 4: all public (flop+turn+river)
      - Channel 5: hole + public
    """
    def __init__(self):
        self.card_tensor = np.zeros((6, 4, 13), dtype=np.float32)
        self.hole_cards = []
        self.public_cards = []
    
    def _mark_card(self, channel, rank, suit):
        self.card_tensor[channel, suit, rank] = 1.0
    
    def set_preflop(self, hole_cards):
        self.hole_cards = hole_cards[:]
        for (r, s) in hole_cards:
            self._mark_card(0, r, s)
            self._mark_card(5, r, s)
    
    def set_flop(self, flop_cards):
        for (r, s) in flop_cards:
            self._mark_card(1, r, s)
            self._mark_card(4, r, s)
            self._mark_card(5, r, s)
        self.public_cards.extend(flop_cards)
    
    def set_turn(self, turn_card):
        if turn_card:
            r, s = turn_card
            self._mark_card(2, r, s)
            self._mark_card(4, r, s)
            self._mark_card(5, r, s)
            self.public_cards.append(turn_card)
    
    def set_river(self, river_card):
        if river_card:
            r, s = river_card
            self._mark_card(3, r, s)
            self._mark_card(4, r, s)
            self._mark_card(5, r, s)
            self.public_cards.append(river_card)
