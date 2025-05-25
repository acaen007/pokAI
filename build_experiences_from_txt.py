import ast
import re
from ai.api.replay import build_replay_experiences
from ai.debug_utils import debug_print

def build_experiences_from_txt(file_path="replay.txt"):
    """
    Reads a replay file and builds training experiences for each hand.
    Each line in the file is expected to be in the format:
        index,final_action_string,hole_cards,board,client_pos,winnings
    For example:
        1,b200c/kk/b200f,['9d', '6h', '3s', '6s'],['Kc', 'Qs'],0,200

    Returns:
        list: A list of experience dictionaries. Each experience dictionary will
              include the replay experience details plus the hand index and winnings.
    """
    all_experiences = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Use regex to extract the six fields:
            # Group 1: hand index (digits)
            # Group 2: action string (no comma)
            # Group 3: hole_cards (string starting with '[' and ending with ']')
            # Group 4: board (same as above)
            # Group 5: client_pos (digits)
            # Group 6: winnings (possibly negative integer)
            pattern = r'^(\d+),([^,]+),(\[.*?\]),(\[.*?\]),(\d+),(-?\d+)$'
            match = re.match(pattern, line)
            if not match:
                debug_print(f"Skipping line due to format mismatch: {line}")
                continue

            hand_index = int(match.group(1))
            action_str = match.group(2)
            try:
                board = ast.literal_eval(match.group(3))
                hole_cards = ast.literal_eval(match.group(4))
            except Exception as e:
                debug_print(f"Error parsing lists in line: {line}\n{e}")
                continue
            client_pos = int(match.group(5))
            winnings = int(match.group(6))

            # Build the replay experiences from this hand
            experiences, cumulative_pot, hero_contrib, villain_contrib = build_replay_experiences(
                action_str, board, hole_cards, client_pos
            )

            # Append hand-level info (winnings and hand index) to each experience
            for exp in experiences:
                exp['hand_index'] = hand_index
                exp['winnings'] = winnings
                all_experiences.append(exp)

    return all_experiences


# Example usage:
if __name__ == '__main__':
    # Assumes that build_replay_experiences (and its dependencies like parse_card)
    # along with constants (NUM_STREETS, BIG_BLIND, SMALL_BLIND) are already defined.
    experiences = build_experiences_from_txt("replay.txt")
    debug_print("Total experiences built:", len(experiences))
    # Optionally, print out a few experiences for inspection
    for exp in experiences[:1]:
        # debug_print("Hand", exp.get('hand_index'), "winnings:", exp.get('winnings'))
        debug_print("Deltas (Hero, Villain):", exp.get('deltas'))
        debug_print("Action Tensor:")
        debug_print(exp.get('action_tensor'))
        debug_print("Card Tensor:")
        debug_print(exp.get('card_tensor'))
        debug_print("-" * 40)
