import random
import torch
from ai.agent import PokerAI
from game.game import PokerGame



def train_ai(num_iterations, batch_size):
    ai_player1 = PokerAI('AI_Player1', stack=1000)
    ai_player2 = PokerAI('AI_Player2', stack=1000)
    game = PokerGame(ai_player1, ai_player2)

    for iteration in range(num_iterations):
        game.start_new_round()
        game_over = False
        while not game_over:
            for player in game.players:
                if isinstance(player, PokerAI):
                    game_state = game.get_game_state(player)
                    action, amount = player.decide_action(game_state)
                    prev_state = player.process_game_state(game_state)
                    game.handle_action(player, action, amount)
                    # Check for game over
                    if game.stage == 'complete' or player.stack <= 0:
                        game_over = True
                        reward = game.pot if player.stack > 0 else -game.pot
                        player.remember(prev_state, amount, reward, prev_state, True)
                        break
                    else:
                        # Reward is zero during the game
                        next_state = player.process_game_state(game.get_game_state(player))
                        player.remember(prev_state, amount, 0, next_state, False)
                else:
                    # If player is not PokerAI, skip or implement logic
                    continue
            if game.players_matched_bets():
                game.next_stage()
        # After the game ends, train the AI agents
        ai_player1.replay(batch_size)
        ai_player2.replay(batch_size)
        # Reset the game for the next iteration
        game.reset()
        # Optionally print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: AI_Player1 stack: {ai_player1.stack}, AI_Player2 stack: {ai_player2.stack}")

if __name__ == "__main__":
    train_ai(num_iterations=10000, batch_size=32)
