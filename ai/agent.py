import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from game.player import Player
import math

class PokerAI(Player):
    def __init__(self, name, stack):
        super().__init__(name, stack)
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = []  # To store experiences for training

    def build_model(self):
        input_size = self.input_size()
        model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        return model

    def input_size(self):
        # Input size calculation
        # Own cards: 13 ranks x 4 suits = 52, two cards => 52 * 2 = 104
        # Community cards: 52 * 5 = 260
        # Pot size: 1
        # Previous actions: Let's consider last 4 actions => 4
        return 104 + 260 + 1 + 4

    def decide_action(self, game_state):
        # Process game state into input tensor
        input_tensor = self.process_game_state(game_state)
        # Forward pass
        with torch.no_grad():
            bet_amount = self.model(input_tensor).item()
        # Round bet amount to the nearest integer
        bet_amount = int(round(bet_amount))
        # Ensure bet amount is within valid range
        bet_amount = max(0, min(bet_amount, self.stack))
        # Decide action based on bet amount
        if bet_amount <= 0:
            if game_state['bets_to_match'] == 0:
                action = 'check'
            else:
                action = 'fold'
            amount = 0
        elif bet_amount >= game_state['bets_to_match']:
            if game_state['bets_to_match'] == 0:
                action = 'bet'
            else:
                action = 'call' if bet_amount == game_state['bets_to_match'] else 'raise'
            amount = bet_amount
        else:
            # Cannot bet less than the required amount to call
            action = 'fold'
            amount = 0
        return action, amount

    def process_game_state(self, game_state):
        # Encode game state into a tensor
        input_vector = []
        # Encode own cards
        own_cards_encoding = self.encode_cards(game_state['player_hand'], 2)
        input_vector.extend(own_cards_encoding)
        # Encode community cards
        community_cards_encoding = self.encode_cards(game_state['community_cards'], 5)
        input_vector.extend(community_cards_encoding)
        # Encode pot size
        pot = game_state['pot'] / 10000.0  # Normalizing
        input_vector.append(pot)
        # Encode previous actions
        actions_encoding = [action / 10.0 for action in game_state['previous_actions'][-4:]]
        actions_encoding += [0] * (4 - len(actions_encoding))  # Pad if less than 4 actions
        input_vector.extend(actions_encoding)
        return torch.tensor(input_vector, dtype=torch.float32)

    def encode_cards(self, cards, num_slots):
        # One-hot encode cards
        encoding = [0] * (52 * num_slots)
        for i in range(num_slots):
            if i < len(cards):
                card = cards[i]
                index = self.card_to_index(card)
                encoding[i * 52 + index] = 1
        return encoding

    def card_to_index(self, card):
        suit_to_index = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}
        rank_to_index = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
                         '7': 5, '8': 6, '9': 7, '10': 8,
                         'J': 9, 'Q': 10, 'K': 11, 'A':12}
        suit = card.suit.lower()
        rank = card.rank
        index = suit_to_index[suit] * 13 + rank_to_index[rank]
        return index

    def remember(self, state, action_value, reward, next_state, done):
        self.memory.append((state, action_value, reward, next_state, done))

    def replay(self, batch_size):
        # Train the model using experiences from memory
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action_value, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + 0.95 * self.model(next_state).item()
            output = self.model(state)
            loss = self.loss_fn(output, torch.tensor([[target]], dtype=torch.float32))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.memory = []
