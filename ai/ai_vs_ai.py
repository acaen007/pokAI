# ai_vs_ai.py

import random
import torch
from game.game import PokerGame
from ai.agent import PokerAI

def train_ai(num_iterations, batch_size):
    ai_player1 = PokerAI('AI_Player1', stack=1000)
    ai_player2 = PokerAI('AI_Player2', stack=1000)
    ai_player1.set_opponent(ai_player2)
    ai_player2.set_opponent(ai_player1)
    game = PokerGame(ai_player1, ai_player2)
    loss_values = []

    for iteration in range(1, num_iterations + 1):
        # Reset stacks if any player has zero chips before starting a new round
        if ai_player1.stack <= 0 or ai_player2.stack <= 0:
            print("A player has run out of chips. Resetting stacks.")
            ai_player1.stack = 1000
            ai_player2.stack = 1000

        game.start_new_round()
        game_over = False

        while not game_over:
            current_player = game.players[game.current_player_index]
            game_state = game.get_game_state(current_player)
            action, amount, action_index = current_player.decide_action(game_state)
            prev_state = current_player.process_game_state(game_state)
            # Get old policy probability
            with torch.no_grad():
                policy, _ = current_player.model(*prev_state)
                old_policy_prob = policy[0, action_index].item()
            game.handle_action(current_player, action, amount, action_index)
            # Move to the next player
            game.current_player_index = (game.current_player_index + 1) % len(game.players)
            # Update the action count
            if current_player.stack > 0:
                game.actions_in_round += 1

            # Check if betting round should end
            if game.stage != 'complete':
                if game.should_move_to_next_stage():
                    if game.stage != 'showdown':
                        game.next_stage()
                    else:
                        game.end_round()
                        game_over = True
                        break  # Exit the loop if the round is over
                    game.reset_actions_in_round()  # Reset after moving to the next stage

            # Check if round is over
            if game.stage == 'complete':
                game_over = True

        # After the game ends, train the AI agents
        loss1 = ai_player1.replay(batch_size)
        loss2 = ai_player2.replay(batch_size)
        if loss1 is not None and loss2 is not None:
            avg_loss = (loss1 + loss2) / 2
            loss_values.append(avg_loss)
        # Optionally print progress
        if iteration % 100 == 0:
            if len(loss_values) > 0:
                avg_loss_over_time = sum(loss_values[-100:]) / len(loss_values[-100:]) if len(loss_values) >= 100 else sum(loss_values) / len(loss_values)
                print(f"Iteration {iteration}, Average Loss: {avg_loss_over_time}")
            else:
                print(f"Iteration {iteration}, No loss data yet.")

    print("Training complete.")
    # Display overall training information
    total_iterations = len(loss_values)
    overall_avg_loss = sum(loss_values) / total_iterations if total_iterations > 0 else 0
    print(f"Total Iterations: {total_iterations}, Overall Average Loss: {overall_avg_loss}")

if __name__ == "__main__":
    train_ai(num_iterations=10, batch_size=32)
