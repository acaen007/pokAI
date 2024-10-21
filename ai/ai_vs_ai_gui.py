import pygame
import sys
import os
import numpy as np
from game.game import PokerGame
from ai.agent import PokerAI
from gui.gui import PokerGUI, SCREEN

class PokerAIGUI(PokerGUI):
    def __init__(self):
        super().__init__()
        self.game = PokerGame(
            PokerAI('AI_Player1', 1000),
            PokerAI('AI_Player2', 1000)
        )
        self.font = pygame.font.SysFont(None, 24)
        self.show_ai_hand = True  # Show both AI hands
        self.running = True
        self.clock = pygame.time.Clock()
        self.human_action_made = False

    def main_loop(self):
        # Start a new round
        self.game.start_new_round()
        while self.running:
            SCREEN.fill((0, 128, 0))  # Green background

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # AI vs AI gameplay
            self.update_game_state()

            # Draw all game elements
            self.draw_game_elements()

            pygame.display.flip()
            self.clock.tick(1)  # Slow down the loop for visualization

        pygame.quit()
        sys.exit()

    def update_game_state(self):
        if self.game.stage in ['pre_flop', 'flop', 'turn', 'river']:
            current_player = self.game.players[self.game.current_player_index]
            game_state = self.game.get_game_state(current_player)
            action, amount = current_player.decide_action(game_state)
            self.game.handle_action(current_player, action, amount)
            # Move to next player
            self.game.current_player_index = (self.game.current_player_index + 1) % 2
            # Check if betting round is over
            if self.players_matched_bets():
                self.game.next_stage()
        elif self.game.stage == 'showdown':
            # Showdown logic
            self.show_ai_hand = True
            self.game.end_round()
            self.game.stage = 'complete'
        elif self.game.stage == 'complete':
            # Start a new round
            if self.game.players[0].stack <= 0 or self.game.players[1].stack <= 0:
                self.running = False  # End the game
                print("Game Over")
            else:
                self.game.start_new_round()
                self.game.current_player_index = 0
                self.show_ai_hand = True

if __name__ == "__main__":
    gui = PokerAIGUI()
    gui.main_loop()
