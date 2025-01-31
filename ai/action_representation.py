# AI/action_representation.py

import numpy as np

class ActionRepresentation:
    """
    Incrementally build a (rounds * max_actions_per_round) x 4 x nb action tensor:
      - rounds * max_actions_per_round channels => e.g., 4 rounds * 6 actions = 24 channels
      - each channel => shape (4, nb), representing [Player1 actions, Player2 actions, Sum of bets, Legal actions] x action options
    """
    def __init__(self, nb=9, max_actions_per_round=6, rounds=4):
        """
        Initializes the ActionRepresentation object.

        Args:
            nb (int): Number of possible actions (e.g., 9).
            max_actions_per_round (int): Maximum number of actions per round (e.g., 6).
            rounds (int): Number of betting rounds (e.g., 4 for Preflop, Flop, Turn, River).
        """
        self.nb = nb
        self.max_actions = max_actions_per_round
        self.rounds = rounds
        # Initialize the action tensor with zeros
        self.action_tensor = np.zeros((rounds * max_actions_per_round, 4, nb), dtype=np.float32)
        self.prev_action = None
        self.sum_row = np.zeros(self.nb)
        # Track the next action index for each round
        self.current_action_index = [0 for _ in range(rounds)]
    
    def get_next_action_index(self, round_id):
        """
        Retrieves the next available action index for a given round.

        Args:
            round_id (int): The current round (0=Preflop, 1=Flop, etc.).

        Returns:
            int: The next action index within the current round.

        Raises:
            ValueError: If the round_id is out of bounds or the maximum number of actions is exceeded.
        """
        if round_id < 0 or round_id >= self.rounds:
            raise ValueError(f"Invalid round_id: {round_id}. Must be between 0 and {self.rounds - 1}.")
        if self.current_action_index[round_id] >= self.max_actions:
            raise ValueError(f"Exceeded maximum actions ({self.max_actions}) for round {round_id}.")
        
        action_index = self.current_action_index[round_id]
        self.current_action_index[round_id] += 1
        return action_index
    
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
            legal_actions = [1, 3, 4, 5, 6, 7, 8] if round_id != 0 else [0, 2, 6, 7, 8]
        if self.prev_action is not None:
            match self.prev_action:
                case 1:
                    legal_actions = [0, 1, 3, 4, 5, 6, 7, 8]
                case 2:
                    legal_actions = [0, 2, 3, 4, 5, 6, 7, 8]
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
        if channel_id >= self.action_tensor.shape[0]:
            raise ValueError(f"Channel ID {channel_id} exceeds tensor dimensions.")
        
        # Set the action for the player
        self.action_tensor[channel_id, player_id, action_idx] = 1.0
        
        # If you want to store sum-of-bets in row=2:
        self.sum_row[action_idx] += 1.0
        self.action_tensor[channel_id, 2, :] = self.sum_row
        
        # If you want to store legal actions in row=3
        for la in legal_actions:
            if 0 <= la < self.nb:
                self.action_tensor[channel_id, 3, la] = 1.0

        self.prev_action = action_idx