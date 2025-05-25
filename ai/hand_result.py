from ai.action_representation import ActionRepresentation
from ai.card_representation import CardRepresentation
import ast
import re
from api.replay import build_replay_experiences
from debug_utils import debug_print

class HandResult:
    def __init__(self, card_reps: list, action_reps: list, actions_taken: list, rewards: list, deltas: list):
        # Fix the length check: compare card_reps, action_reps and rewards.
        if len(card_reps) != len(action_reps) or len(card_reps) != len(rewards):
            debug_print(len(card_reps), len(action_reps), len(rewards))
            raise ValueError("card_reps, rewards and action_reps must have the same length")
        # Here we assume that card_reps and action_reps already contain the numpy arrays.
        self.states = list((card_rep, action_rep) for card_rep, action_rep in zip(card_reps, action_reps))
        self.rewards = rewards
        self.rounds = [{
            'state': self.states[i],
            'action_taken': actions_taken[i],
            'reward': rewards[i],
            'deltas': deltas[i]
        } for i in range(len(self.states))]

    def new_state(self, card_rep: CardRepresentation, action_rep: ActionRepresentation, action_taken: int, reward: int, deltas: list):
        # If card_rep or action_rep are objects, try to use their .card_tensor or .action_tensor attributes; otherwise, use them directly.
        if hasattr(card_rep, "card_tensor"):
            card_array = card_rep.card_tensor
        else:
            card_array = card_rep
        if hasattr(action_rep, "action_tensor"):
            action_array = action_rep.action_tensor
        else:
            action_array = action_rep
        self.states.append((card_array, action_array))
        self.rewards.append(reward)
        self.rounds.append({
            'state': (card_array, action_array),
            'action_taken': action_taken,
            'reward': reward,
            'deltas': deltas
        })

def create_hands_from_experiences(experiences: list) -> list:
    # Separate experiences by hand_index field.
    hands = {}
    for exp in experiences:
        hand_index = exp['hand_index']
        if hand_index not in hands:
            hands[hand_index] = []
        hands[hand_index].append(exp)
    # Create HandResult objects from the experiences.
    hand_results = []
    for hand_index, hand_exps in hands.items():
        card_reps = []
        action_reps = []
        actions_taken = []
        rewards = []
        deltas = []
        for exp in hand_exps:
            card_reps.append(exp['card_tensor'])
            action_reps.append(exp['action_tensor'])
            actions_taken.append(exp['action_idx'])
            rewards.append(exp['reward'])
            deltas.append(exp['deltas'])
        hand_result = HandResult(card_reps, action_reps, actions_taken, rewards, deltas)
        hand_results.append(hand_result)
    return hand_results

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

            experiences_list, cumulative_pot, hero_contrib, villain_contrib = build_replay_experiences(
                action_str, board, hole_cards, client_pos
            )
            for exp in experiences_list:
                exp['hand_index'] = hand_index
                exp['winnings'] = winnings
                all_experiences.append(exp)

    return all_experiences
