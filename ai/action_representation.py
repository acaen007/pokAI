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
        self.prev_action = None
        self.sum_row = np.zeros(self.nb)
    
    def add_action(self, round_id, action_index_in_round, player_id, action_idx):
        """
        round_id in [0..3]
        action_index_in_round in [0..5]
        player_id in [0..1]  (hero vs villain)
        action_idx in [0..nb-1]
        """
        # Here I define the legal action depending on the action of the other player. Also I initialize the sum row
        if action_index_in_round == 0:
            self.prev_action = None
            self.sum_row = np.zeros(self.nb)
            legal_actions = range(self.nb)
        if self.prev_action is not None:
            match self.prev_action:
                case 1:
                    legal_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                case 3:
                    legal_actions = [0, 2, 3, 4, 5, 6, 7, 8] # Calling would mean betting 1/3 pot < 1/2 pot
                case 4:
                    legal_actions = [0, 2, 3, 4, 5, 6, 7, 8] # Calling would mean betting 3/7 pot < 1/2 pot
                case 5:
                    legal_actions = [0, 2, 4, 5, 6, 7, 8] # Calling would mean betting 1/2 pot < 3/4 pot
                case 6:
                    legal_actions = [0, 2, 4, 5, 6, 7, 8] # Calling would mean betting 3/5 pot < 3/4 pot
                case 7:
                    legal_actions = [0, 2, 4, 5, 6, 7, 8] # Calling would mean betting 2/3 pot < 3/4 pot
                case 8:
                    legal_actions = [0, 2] # Against an all-in, only call or fold
                case _:
                    raise ValueError(f"Unknown action: {self.prev_action}. Folds and calls should not get to this point")

        channel_id = round_id * self.max_actions + action_index_in_round
        self.action_tensor[channel_id, player_id, action_idx] = 1.0
        
        # If you want to store sum-of-bets in row=2:
        self.sum_row[action_idx] += 1.0
        self.action_tensor[channel_id, 2, :] = self.sum_row
        
        # If you want to store legal actions in row=3
        for la in legal_actions:
            if 0 <= la < self.nb:
                self.action_tensor[channel_id, 3, la] = 1.0

        self.prev_action = action_idx