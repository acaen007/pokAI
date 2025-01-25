# agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from game.player import Player

class PokerAIModel(nn.Module):
    def __init__(self, nb_actions):
        super(PokerAIModel, self).__init__()
        self.nb_actions = nb_actions
        self.card_embedding = nn.Linear(52, 128)
        self.action_embedding = nn.Linear(nb_actions, 128)
        self.fc1 = nn.Linear(256, 256)
        self.fc_policy = nn.Linear(256, nb_actions)
        self.fc_value = nn.Linear(256, 1)
        self.apply(self._init_weights)  # Initialize weights

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight, -0.1, 0.1)
            nn.init.constant_(module.bias, 0)

    def forward(self, card_tensor, action_tensor):
        card_embed = torch.relu(self.card_embedding(card_tensor))
        action_embed = torch.relu(self.action_embedding(action_tensor))
        combined = torch.cat((card_embed, action_embed), dim=1)
        x = torch.relu(self.fc1(combined))
        policy = torch.softmax(self.fc_policy(x), dim=1)
        value = self.fc_value(x)
        return policy, value

class PokerAI(Player):
    def __init__(self, name, stack):
        super().__init__(name, stack)
        self.betting_options = [
            {'name': 'fold', 'type': 'fold'},
            {'name': 'check', 'type': 'check'},
            {'name': 'call', 'type': 'call'},
            {'name': 'bet_0.5_pot', 'type': 'bet', 'multiplier': 0.5},
            {'name': 'bet_0.75_pot', 'type': 'bet', 'multiplier': 0.75},
            {'name': 'bet_pot', 'type': 'bet', 'multiplier': 1},
            {'name': 'bet_1.5_pot', 'type': 'bet', 'multiplier': 1.5},
            {'name': 'bet_2_pot', 'type': 'bet', 'multiplier': 2},
            {'name': 'all_in', 'type': 'bet', 'multiplier': None}
        ]
        self.nb = len(self.betting_options)
        self.model = PokerAIModel(self.nb)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()
        self.memory = []
        self.epsilon = 0.5  # Increased exploration rate
        self.opponent = None  # Will be set later
        self.current_bet = 0  # Initialize current bet
        self.minimum_bet_amount = 10  # Set a minimum bet amount

    def set_opponent(self, opponent):
        self.opponent = opponent

    def process_game_state(self, game_state):
        # Convert game state into tensors
        card_tensor = self.cards_to_tensor(self.hand.cards + game_state['community_cards'])
        action_history_tensor = self.actions_to_tensor(game_state['action_history'])
        return card_tensor, action_history_tensor

    def cards_to_tensor(self, cards):
        tensor = torch.zeros(52)
        for card in cards:
            index = self.card_to_index(card)
            tensor[index] = 1
        return tensor.unsqueeze(0)  # Add batch dimension

    def card_to_index(self, card):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suit_index = suits.index(card.suit)
        rank_index = ranks.index(card.rank)
        return suit_index * 13 + rank_index

    def actions_to_tensor(self, action_history):
        tensor = torch.zeros(self.nb)
        for action in action_history:
            action_index = action['action_index']
            tensor[action_index] += 1
        return tensor.unsqueeze(0)  # Add batch dimension

    def decide_action(self, game_state):
        # Check if betting is allowed
        if game_state.get('betting_allowed', True):
            card_tensor, action_tensor = self.process_game_state(game_state)
            with torch.no_grad():
                policy, _ = self.model(card_tensor, action_tensor)
            # Epsilon-greedy exploration
            legal_actions = self.get_legal_actions(game_state)
            if random.random() < self.epsilon:
                action_index = random.choice(legal_actions)
            else:
                # Select action with highest probability among legal actions
                policy_probs = policy.numpy().flatten()
                policy_probs = [policy_probs[i] if i in legal_actions else 0 for i in range(self.nb)]
                total_prob = np.sum(policy_probs)
                if total_prob == 0:
                    # All legal actions have zero probability, choose randomly
                    action_index = random.choice(legal_actions)
                else:
                    policy_probs = [p / total_prob for p in policy_probs]
                    action_index = np.random.choice(self.nb, p=policy_probs)
            # Map action index to actual action and amount
            action, amount = self.map_action_index_to_action(action_index, game_state)
        else:
            # If betting is not allowed, the only valid action is 'check'
            action = 'check'
            amount = 0
            action_index = 1  # 'check' action index
        return action, amount, action_index

    def map_action_index_to_action(self, action_index, game_state):
        option = self.betting_options[action_index]
        pot = game_state['pot']
        bets_to_match = game_state['bets_to_match']
        current_bet = game_state['current_bet']
        player_stack = game_state['player_stack']

        if option['type'] == 'fold':
            action = 'fold'
            total_bet = current_bet
        elif option['type'] == 'check':
            action = 'check'
            total_bet = current_bet
        elif option['type'] == 'call':
            action = 'call'
            required_call = bets_to_match - current_bet
            total_bet = min(bets_to_match, current_bet + player_stack)
        elif option['type'] == 'bet':
            if option['name'] == 'all_in':
                bet_amount = player_stack
            else:
                bet_amount = max(int(pot * option['multiplier']), self.minimum_bet_amount)
                bet_amount = min(bet_amount, player_stack)
            if bets_to_match == current_bet:
                action = 'bet'
                total_bet = current_bet + bet_amount
            else:
                action = 'raise'
                min_raise = bets_to_match - current_bet
                bet_amount = max(bet_amount, min_raise)
                bet_amount = min(bet_amount, player_stack)
                total_bet = current_bet + bet_amount
        else:
            action = 'fold'
            total_bet = current_bet
        return action, total_bet

    def get_legal_actions(self, game_state):
        legal_actions = []
        bets_to_match = game_state['bets_to_match']
        current_bet = game_state['current_bet']
        player_stack = game_state['player_stack']

        if bets_to_match > current_bet:
            # Outstanding bet: can fold, call, or raise
            legal_actions.append(0)  # 'fold' action index
            if player_stack > 0:
                legal_actions.append(2)  # 'call' action index
                for idx, option in enumerate(self.betting_options[3:], start=3):
                    legal_actions.append(idx)  # Include raise options
        else:
            # No outstanding bet: can check or bet
            legal_actions.append(1)  # 'check' action index
            if player_stack > 0:
                for idx, option in enumerate(self.betting_options[3:], start=3):
                    legal_actions.append(idx)  # Include bet options

        return legal_actions

    def remember(self, state, action, reward, next_state, done, old_policy_prob):
        self.memory.append((state, action, reward, next_state, done, old_policy_prob))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done, old_policy_prob in minibatch:
            card_tensor, action_tensor = state
            policy, value = self.model(card_tensor, action_tensor)
            target = reward
            if not done:
                with torch.no_grad():
                    next_policy, next_value = self.model(*next_state)
                    target = reward + 0.99 * next_value.item()
            advantage = target - value.item()
            # Compute policy loss (using policy gradient)
            policy_loss = -torch.log(policy[0, action]) * advantage
            # Compute value loss
            value_loss = self.loss_fn(value, torch.tensor([[target]]))
            # Total loss
            loss = policy_loss + value_loss
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # Clear memory after training
        self.memory = []
        return loss.item()
