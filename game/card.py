class Card:
    SUITS = ['hearts', 'diamonds', 'clubs', 'spades']
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return f"{self.rank} of {self.suit}"
    
    @staticmethod
    def from_string(card_str):
        rank_char = card_str[:-1]
        suit_char = card_str[-1]
        rank = rank_char.upper()
        suit = suit_char.lower()
        if rank and suit:
            return Card(suit, rank)
        else:
            raise ValueError(f"Invalid card string: {card_str}")
