import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

NUM_STREETS = 4
SMALL_BLIND = 50
BIG_BLIND = 100

RANKS = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
    '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
}

SUITS = {
    's': 0,  # Spades
    'c': 1,  # Clubs
    'd': 2,  # Diamonds
    'h': 3   # Hearts
}

def parse_card(card_str):
    """
    Parses a 2-character card string into (rank, suit) integers.

    Args:
        card_str (str): e.g., '8s', '4c', 'Ts', 'Ah'

    Returns:
        tuple: (rank_int, suit_int)

    Raises:
        ValueError: If the card string is invalid.
    """
    if len(card_str) != 2:
        raise ValueError(f"Invalid card string length: '{card_str}'. Expected 2 characters.")
    
    rank_str, suit_str = card_str[0], card_str[1]

    rank = RANKS.get(rank_str.upper())
    suit = SUITS.get(suit_str.lower())

    if rank is None:
        raise ValueError(f"Invalid rank '{rank_str}' in card string: '{card_str}'.")
    if suit is None:
        raise ValueError(f"Invalid suit '{suit_str}' in card string: '{card_str}'.")

    return (rank, suit)


def build_replay_experiences(action_str, board, hole_cards, client_pos):
    """
    Parses the action string, board, and hole cards to build replay experiences.
    
    Args:
        action_str (str): Actions taken, e.g., "b200c/kk/kk/kb200c"
        board (list): Community cards, e.g., ['Kc', 'Th', '4s', 'Ts']
        hole_cards (list): Hero's hole cards, e.g., ['As', '2d']
        client_pos (int): Hero's seat, 0 for BB, 1 for SB
    
    Returns:
        tuple: (list of experiences, updated_cumulative_pot, updated_hero_contrib, updated_villain_contrib)
               Each hero experience now includes a 'reward' key that equals the negative of the amount
               the hero just bet (or called). For non-bet/check/fold actions, reward is 0.
    """
    from card_representation import CardRepresentation
    from action_representation import ActionRepresentation

    # Initialize representations
    card_rep = CardRepresentation()
    action_rep = ActionRepresentation(nb=9, max_actions_per_round=6, rounds=4)

    # Initialize player contributions based on fixed seats
    # Seat 0: BB, Seat 1: SB
    player_contrib = {0: BIG_BLIND, 1: SMALL_BLIND}

    # Define hero and villain positions based on client_pos
    if client_pos == 0:
        hero_pos = 0  # Hero is in BB
        villain_pos = 1  # Villain is in SB
    else:
        hero_pos = 1  # Hero is in SB
        villain_pos = 0  # Villain is in BB

    # Initialize cumulative pot
    cumulative_pot = player_contrib[0] + player_contrib[1]

    # Set preflop hole cards
    if hole_cards:
        try:
            parsed_hole_cards = [parse_card(card) for card in hole_cards]
            card_rep.set_preflop(parsed_hole_cards)
        except ValueError as ve:
            print(f"Error parsing hole cards: {ve}")
            return [], cumulative_pot, player_contrib[hero_pos], player_contrib[villain_pos]

    # Split action_str by '/' to get actions per street
    streets = action_str.split('/')

    # Assign board cards to streets
    street_boards = {
        0: [],  # Preflop
        1: board[:3] if len(board) >= 3 else [],
        2: board[:4] if len(board) >= 4 else [],
        3: board[:5] if len(board) >= 5 else []
    }

    experiences = []

    for st in range(NUM_STREETS):
        street_actions = streets[st] if st < len(streets) else ''
        print(f"\nProcessing Street {st}: Actions='{street_actions}'")

        # Set board cards if any
        if st > 0 and st < NUM_STREETS and street_boards[st]:
            try:
                if st == 1:
                    parsed_flop = [parse_card(card) for card in street_boards[st]]
                    card_rep.set_flop(parsed_flop)
                elif st == 2:
                    parsed_turn = parse_card(street_boards[st][3])
                    card_rep.set_turn(parsed_turn)
                elif st == 3:
                    parsed_river = parse_card(street_boards[st][4])
                    card_rep.set_river(parsed_river)
            except ValueError as ve:
                print(f"Error parsing board cards: {ve}")
                return experiences, cumulative_pot, player_contrib[hero_pos], player_contrib[villain_pos]

        # Determine who acts first in this street
        if st == 0:
            # Preflop: Small Blind (seat 1) acts first
            first_actor = 1
            print(f"First actor in Street {st}: Small blind (Seat {first_actor})")
        else:
            # Postflop: Big Blind (seat 0) acts first
            first_actor = 0
            print(f"First actor in Street {st}: Big Blind (Seat {first_actor})")

        pos = first_actor

        # Initialize current bet for the street
        current_bet = max(player_contrib.values())
        print(f"Initial contributions: Hero (Seat {hero_pos}) = {player_contrib[hero_pos]}, "
              f"Villain (Seat {villain_pos}) = {player_contrib[villain_pos]}")
        print(f"Current bet: {current_bet}")

        # Parse actions in this street
        i = 0
        while i < len(street_actions):
            c = street_actions[i]
            if c in ['f', 'k', 'c', 'b', 'r']:
                if c in ['f', 'k', 'c']:
                    action = c
                    i += 1
                elif c in ['b', 'r']:
                    # Extract the full bet or raise action, e.g., 'b100' or 'r200'
                    j = i + 1
                    while j < len(street_actions) and street_actions[j].isdigit():
                        j += 1
                    action = street_actions[i:j]
                    i = j
            else:
                print(f"Unexpected character '{c}' in action string.")
                i += 1
                continue

            # Map action to action_idx and update contributions
            if action == 'f':
                action_idx = 0
                # No money added for a fold.
                added_amount = 0
            elif action == 'k':
                action_idx = 1
                # No money added for a check.
                added_amount = 0
            elif action == 'c':
                action_idx = 2
                # Calculate the amount to call.
                to_call = (player_contrib[villain_pos] - player_contrib[pos]) if pos == hero_pos else (player_contrib[hero_pos] - player_contrib[pos])
                added_amount = to_call if to_call > 0 else 0
                if added_amount > 0:
                    player_contrib[pos] += added_amount
                    cumulative_pot += added_amount
                print(f"Action: 'c' by {'Hero' if pos == hero_pos else 'Villain'} (Seat {pos}) - To Call: {added_amount}")
            elif action.startswith('b'):
                try:
                    bet_amount = int(action[1:])
                except ValueError:
                    print(f"Invalid bet amount in action '{action}'. Skipping.")
                    continue

                # Compute effective bet for preflop: subtract the blind already contributed.
                if st == 0:
                    if pos == 1:
                        effective_bet = bet_amount - SMALL_BLIND
                    else:  # pos == 0
                        effective_bet = bet_amount - BIG_BLIND
                else:
                    effective_bet = bet_amount

                added_amount = effective_bet
                pot_fraction = effective_bet / cumulative_pot if cumulative_pot > 0 else 0
                if pot_fraction <= 0.5:
                    action_idx = 3
                elif pot_fraction <= 0.75:
                    action_idx = 4
                elif pot_fraction == 1.0:
                    action_idx = 5
                elif pot_fraction <= 1.5:
                    action_idx = 6
                elif pot_fraction <= 2.0:
                    action_idx = 7
                else:
                    action_idx = 8

                # Update contributions using the effective bet (i.e. bet beyond the blind)
                player_contrib[pos] += effective_bet
                cumulative_pot += effective_bet

                # Update current_bet to reflect the total contribution after the bet
                current_bet = player_contrib[pos]
                print(f"Action: '{action}' by {'Hero' if pos == hero_pos else 'Villain'} (Seat {pos}) - Effective Bet: {effective_bet}")
            else:
                # Default to check if action unrecognized
                action_idx = 1
                added_amount = 0

            # Get the next action index for the current round
            try:
                action_index_in_round = action_rep.get_next_action_index(st)
            except ValueError as ve:
                print(f"Error: {ve}")
                continue

            # Always add hero's actions on row 0 and villain's on row 1
            row = 0 if pos == hero_pos else 1
            action_rep.add_action(st, action_index_in_round, row, action_idx)

            # If this is hero's action, record the experience with the reward.
            # The reward is defined as the negative of the amount just bet (or called).
            if pos == hero_pos:
                if action.startswith('b'):
                    reward = -added_amount
                elif action == 'c':
                    reward = -added_amount
                else:
                    reward = 0

                card_tensor_copy = card_rep.card_tensor.copy()
                action_tensor_copy = action_rep.action_tensor.copy()
                experience = {
                    'card_tensor': card_tensor_copy,
                    'action_tensor': action_tensor_copy,
                    'action_idx': action_idx,
                    'deltas': (3, player_contrib[hero_pos], player_contrib[villain_pos]),
                    'reward': reward
                }
                experiences.append(experience)
                print(f"Recorded experience for Hero at Street {st}: Action='{action}', Action_idx={action_idx}, Reward={reward}")
            else:
                print(f"Action by Villain at Street {st}: Action='{action}', Action_idx={action_idx} (No experience recorded)")

            # Switch player
            pos = 1 - pos

    # --- Add final state experience ---
    # This final state captures the hand's state even if the last action wasn't by the hero.
    card_tensor_copy = card_rep.card_tensor.copy()
    action_tensor_copy = action_rep.action_tensor.copy()
    final_experience = {
        'card_tensor': card_tensor_copy,
        'action_tensor': action_tensor_copy,
        'action_idx': -1,  # Marker indicating this is the final state
        'deltas': (player_contrib[hero_pos], player_contrib[villain_pos]),
        'reward': 0
    }
    experiences.append(final_experience)
    print("Recorded final state experience.")

    return experiences, cumulative_pot, player_contrib[hero_pos], player_contrib[villain_pos]


    
def test_build_replay_experiences_suite():
    test_cases = [
        # {
        #     'description': 'Simple Preflop Call',
        #     'action_str': "ck/kk/kk/kk",
        #     'board': ['3h', '4d', '5s', '6c', '7h'],
        #     'hole_cards': ['As', '2d'],
        #     'client_pos': 1,  # 1: Small Blind 0: Big Blind
        #     'expected_hero_contrib': SMALL_BLIND + 50,  # 50 + 50 = 100
        #     'expected_villain_contrib': BIG_BLIND,      # 100
        #     'expected_cumulative_pot': 200,
        #     'expected_experiences': 3
        # },
        # {
        #     'description': 'Hero Raises Preflop, Villain Folds',
        #     'action_str': "b200f/",
        #     'board': [],
        #     'hole_cards': ['As', 'Kd'],
        #     'client_pos': 0,  
        #     'expected_hero_contrib': BIG_BLIND,  # 50 + 150 = 200
        #     'expected_villain_contrib': SMALL_BLIND + 150,        # 100
        #     'expected_cumulative_pot': 300,
        #     'expected_experiences': 1
        # },
        {
            'description': 'Multiple Bets and Calls Across Streets',
            'action_str': "ck/kk/b100c/b200c",
            'board': ['Kc', 'Th', '4s', 'Ts', '9d'],
            'hole_cards': ['As', '2d'],
            'client_pos': 0,  # Small Blind
            'expected_hero_contrib': SMALL_BLIND + 50 + 100 + 200,  # 50 + 50 + 100 + 200 = 400
            'expected_villain_contrib': BIG_BLIND + 100 + 200,      # 100 + 100 + 200 = 400
            'expected_cumulative_pot': 800,
            'expected_experiences': 5
        },
        # {
        #     'description': 'All-In Scenario',
        #     'action_str': "b20000f",
        #     'board': [],
        #     'hole_cards': ['Ah', 'Ad'],
        #     'client_pos': 0,  # Small Blind
        #     'expected_hero_contrib': BIG_BLIND,  # 50 + 500 = 550
        #     'expected_villain_contrib': 20000,  # 100 + 500 = 600
        #     'expected_cumulative_pot': 20100,
        #     'expected_experiences': 1
        # },
        # {
        #     'description': 'Villain Raises and Hero Re-Raises',
        #     'action_str': "b200b400f/",
        #     'board': [],
        #     'hole_cards': ['Qs', 'Jh'],
        #     'client_pos': 1,  # Small Blind
        #     'expected_hero_contrib': 200,  
        #     'expected_villain_contrib': 400,       
        #     'expected_cumulative_pot': 600
        # },
        # {
        #     'description': 'Hero Folds After Villain Bets',
        #     'action_str': "ck/kk/kk/b100f",
        #     'board': ['Kc', 'Th', '4s', 'Ts', '9d'],
        #     'hole_cards': ['Js', '3d'],
        #     'client_pos': 1,  # Small Blind
        #     'expected_hero_contrib': SMALL_BLIND + 50,  # 50 + 50 = 100
        #     'expected_villain_contrib': BIG_BLIND + 100, # 100 + 100 = 200
        #     'expected_cumulative_pot': 300
        # },
    ]
    
    for idx, test in enumerate(test_cases, 1):
        print(f"Running Test Case {idx}: {test['description']}")
        experiences, cumulative_pot, hero_contrib, villain_contrib = build_replay_experiences(
            test['action_str'],
            test['board'],
            test['hole_cards'],
            test['client_pos']
        )
    
        print(f"Hero Contribution: {hero_contrib}, Expected: {test['expected_hero_contrib']}")
        print(f"Villain Contribution: {villain_contrib}, Expected: {test['expected_villain_contrib']}")
        print(f"Cumulative Pot: {cumulative_pot}, Expected: {test['expected_cumulative_pot']}")
    
        assert len(experiences) == test.get('expected_experiences', None) or len(experiences) == 4, \
            f"Expected {test.get('expected_experiences', 4)} experiences, got {len(experiences)}"
        assert hero_contrib == test['expected_hero_contrib'], \
            f"Hero contrib mismatch: {hero_contrib} != {test['expected_hero_contrib']}"
        assert villain_contrib == test['expected_villain_contrib'], \
            f"Villain contrib mismatch: {villain_contrib} != {test['expected_villain_contrib']}"
        assert cumulative_pot == test['expected_cumulative_pot'], \
            f"Cumulative pot mismatch: {cumulative_pot} != {test['expected_cumulative_pot']}"
        print("Test Passed.\n")

        for i, exp in enumerate(experiences):
            print(f"Experience {i}:")
            print(f"Delta1: {exp['deltas'][0]}, Delta2: {exp['deltas'][1]}")
            print(f"Card Tensor:\n{exp['card_tensor']}")
            print(f"Action Tensor:\n{exp['action_tensor']}")
            print(f"Action Index: {exp['action_idx']}")
            print()
    
    print("All test cases passed successfully.")
    
test_build_replay_experiences_suite()



