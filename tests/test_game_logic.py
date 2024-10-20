import unittest
from game.card import Card
from game.deck import Deck
from game.player import Player
from game.game import PokerGame

class TestPokerGame(unittest.TestCase):
    def test_card_creation(self):
        card = Card('Hearts', 'A')
        self.assertEqual(str(card), 'A of Hearts')

    def test_deck(self):
        deck = Deck()
        self.assertEqual(len(deck.cards), 52)
        card = deck.deal()
        self.assertEqual(len(deck.cards), 51)

    # Add more tests for each component

if __name__ == '__main__':
    unittest.main()
