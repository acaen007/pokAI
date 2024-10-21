import unittest
from game.card import Card
from game.hand_evaluator import evaluate_hand

class TestHandEvaluator(unittest.TestCase):
    def test_high_card(self):
        hand = [Card('hearts', 'A'), Card('spades', 'K')]
        community = [Card('diamonds', '2'), Card('clubs', '5'), Card('spades', '9'), Card('hearts', 'J'), Card('diamonds', '7')]
        rank, high_cards = evaluate_hand(hand, community)
        self.assertEqual(rank, 0)
        self.assertEqual(high_cards, [14, 13, 11, 9, 7])

    def test_one_pair(self):
        hand = [Card('hearts', 'A'), Card('spades', 'A')]
        community = [Card('diamonds', '2'), Card('clubs', '5'), Card('spades', '9'), Card('hearts', 'J'), Card('diamonds', '7')]
        rank, high_cards = evaluate_hand(hand, community)
        self.assertEqual(rank, 1)
        self.assertEqual(high_cards[0], 14)  # Pair of Aces

    def test_two_pair(self):
        hand = [Card('hearts', 'A'), Card('spades', 'A')]
        community = [Card('diamonds', '2'), Card('clubs', '2'), Card('spades', '9'), Card('hearts', 'J'), Card('diamonds', '7')]
        rank, high_cards = evaluate_hand(hand, community)
        self.assertEqual(rank, 2)
        self.assertEqual(high_cards[:2], [14, 2])  # Pairs of Aces and Twos

    # Add more tests for each hand type

if __name__ == '__main__':
    unittest.main()
