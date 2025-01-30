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
    
    def add_action(self, round_id, action_index_in_round, player_id, action_idx, legal_actions=None, sum_idx=None):
        """
        Adds an action to the action tensor.

        Args:
            round_id (int): The current round (0=Preflop, 1=Flop, etc.).
            action_index_in_round (int): The action index within the current round (0 to max_actions_per_round-1).
            player_id (int): Player identifier (0 or 1).
            action_idx (int): Action index (0 to nb-1).
            legal_actions (list, optional): List of action indices that are legal in the current state.
            sum_idx (int, optional): Index to mark the sum of bets row.
        """
        channel_id = round_id * self.max_actions + action_index_in_round
        if channel_id >= self.action_tensor.shape[0]:
            raise ValueError(f"Channel ID {channel_id} exceeds tensor dimensions.")
        
        # Set the action for the player
        self.action_tensor[channel_id, player_id, action_idx] = 1.0
        
        # Optionally set the sum of bets row
        if sum_idx is not None and 0 <= sum_idx < self.nb:
            self.action_tensor[channel_id, 2, sum_idx] = 1.0
        
        # Optionally set the legal actions row
        if legal_actions:
            for la in legal_actions:
                if 0 <= la < self.nb:
                    self.action_tensor[channel_id, 3, la] = 1.0
