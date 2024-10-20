import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game.player import Player

class PokerAI(Player):
    def __init__(self, name, stack):
        super().__init__(name, stack)
        self.model = self.build_model()

    def build_model(self):
        # Simple feedforward network as a placeholder
        model = nn.Sequential(
            nn.Linear(self.input_size(), 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Outputs for fold, call, raise
        )
        return model

    def input_size(self):
        # Define the size of the input layer
        # For example: own cards + community cards + other game state info
        return 52 + 52 + 10  # Placeholder sizes

    def decide_action(self, game_state):
        # Convert game_state to tensor
        state_tensor = torch.tensor(game_state, dtype=torch.float32)
        # Forward pass through the model
        with torch.no_grad():
            action_probs = self.model(state_tensor)
        # Decide action based on output
        action = torch.argmax(action_probs).item()
        return action  # Return action index

    def bet(self, amount):
        # Override bet function if necessary
        return super().bet(amount)
