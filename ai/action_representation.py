# AI/action_representation.py

import numpy as np

class ActionRepresentation:
    """
    Incrementally build a 24 x 4 x nb action tensor:
      - 24 channels => 4 rounds * 6 actions per round
      - each channel => shape (4, nb), for [P1 row, P2 row, sum row, legal row] x bet options
    """
    def __init__(self, nb=9, max_actions_per_round=6, rounds=4):
        self.nb = nb
        self.max_actions = max_actions_per_round
        self.rounds = rounds
        # 24 channels total, each is 4 x nb (in this case 4 x 9)
        self.action_tensor = np.zeros((rounds * max_actions_per_round, 4, nb), dtype=np.float32)
    
    def add_action(self, round_id, action_index_in_round, player_id, action_idx, legal_actions=None, sum_idx=None):
        """
        round_id in [0..3]
        action_index_in_round in [0..5]
        player_id in [0..1]  (hero vs villain)
        action_idx in [0..nb-1]
        legal_actions: list of valid action_idx's
        sum_idx: optional single int to mark row=2 (the 'sum of bets' row)
        """
        channel_id = round_id * self.max_actions + action_index_in_round
        self.action_tensor[channel_id, player_id, action_idx] = 1.0
        
        # If you want to store sum-of-bets in row=2:
        if sum_idx is not None and 0 <= sum_idx < self.nb:
            self.action_tensor[channel_id, 2, sum_idx] = 1.0
        
        # If you want to store legal actions in row=3
        if legal_actions:
            for la in legal_actions:
                if 0 <= la < self.nb:
                    self.action_tensor[channel_id, 3, la] = 1.0
