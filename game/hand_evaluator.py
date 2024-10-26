# hand_evaluator.py
from collections import Counter
from itertools import combinations

def evaluate_hand(hand_cards, community_cards):
    """
    Evaluate the best poker hand given a player's hand and the community cards.
    Returns a tuple (hand_rank, high_cards), where:
    - hand_rank is an integer (higher is better)
    - high_cards is a list of card ranks used to break ties
    """
    cards = hand_cards + community_cards
    all_combinations = get_all_five_card_combinations(cards)
    best_rank = -1
    best_high_cards = []
    for combo in all_combinations:
        rank, high_cards = classify_hand(combo)
        if rank > best_rank or (rank == best_rank and high_cards > best_high_cards):
            best_rank = rank
            best_high_cards = high_cards
    return best_rank, best_high_cards

def get_all_five_card_combinations(cards):
    return list(combinations(cards, 5))

def classify_hand(cards):
    ranks = [card.rank for card in cards]
    suits = [card.suit for card in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    sorted_ranks = sorted([rank_to_value(rank) for rank in ranks], reverse=True)

    is_flush = max(suit_counts.values()) == 5
    is_straight, straight_high = check_straight(sorted_ranks)

    if is_flush and is_straight:
        return (8, [straight_high])  # Straight Flush
    elif 4 in rank_counts.values():
        four_kind_rank = get_key_by_value(rank_counts, 4)
        high_card = max([rank_to_value(rank) for rank in ranks if rank_to_value(rank) != rank_to_value(four_kind_rank)])
        return (7, [rank_to_value(four_kind_rank), high_card])  # Four of a Kind
    elif 3 in rank_counts.values() and 2 in rank_counts.values():
        three_kind_rank = get_key_by_value(rank_counts, 3)
        pair_rank = get_key_by_value(rank_counts, 2)
        return (6, [rank_to_value(three_kind_rank), rank_to_value(pair_rank)])  # Full House
    elif is_flush:
        return (5, sorted_ranks)  # Flush
    elif is_straight:
        return (4, [straight_high])  # Straight
    elif 3 in rank_counts.values():
        three_kind_rank = get_key_by_value(rank_counts, 3)
        kickers = get_kickers(rank_counts, exclude=[three_kind_rank], count=2)
        return (3, [rank_to_value(three_kind_rank)] + kickers)  # Three of a Kind
    elif list(rank_counts.values()).count(2) == 2:
        pairs = get_keys_by_value(rank_counts, 2)
        pairs_sorted = sorted([rank_to_value(rank) for rank in pairs], reverse=True)
        kicker = get_kickers(rank_counts, exclude=pairs, count=1)[0]
        return (2, pairs_sorted + [kicker])  # Two Pair
    elif 2 in rank_counts.values():
        pair_rank = get_key_by_value(rank_counts, 2)
        kickers = get_kickers(rank_counts, exclude=[pair_rank], count=3)
        return (1, [rank_to_value(pair_rank)] + kickers)  # One Pair
    else:
        return (0, sorted_ranks[:5])  # High Card

def rank_to_value(rank):
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                   '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    return rank_values[rank]

def check_straight(sorted_ranks):
    unique_ranks = sorted(set(sorted_ranks), reverse=True)
    for i in range(len(unique_ranks) - 4):
        window = unique_ranks[i:i+5]
        if window[0] - window[4] == 4:
            return True, window[0]
    # Check for wheel straight (A-2-3-4-5)
    if set([14, 2, 3, 4, 5]).issubset(set(sorted_ranks)):
        return True, 5
    return False, None

def get_key_by_value(counter, value):
    for key, val in counter.items():
        if val == value:
            return key
    return None

def get_keys_by_value(counter, value):
    return [key for key, val in counter.items() if val == value]

def get_kickers(rank_counts, exclude, count):
    kickers = [rank_to_value(rank) for rank in rank_counts.keys() if rank not in exclude]
    kickers_sorted = sorted(kickers, reverse=True)
    return kickers_sorted[:count]

def get_hand_rank_name(rank):
    rank_names = {
        8: 'Straight Flush',
        7: 'Four of a Kind',
        6: 'Full House',
        5: 'Flush',
        4: 'Straight',
        3: 'Three of a Kind',
        2: 'Two Pair',
        1: 'One Pair',
        0: 'High Card',
    }
    return rank_names.get(rank, 'Unknown Hand')

