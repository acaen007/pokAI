# AI/toy_experiment.py

import os
import sys
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.api.replay import BIG_BLIND, SMALL_BLIND
from ppo_utils import (
    a_gae,
    tc_loss_function,
    ratio,
    r_gamma,
    v_loss,
    get_action_from_probs,
    make_model_value_function
)
from card_representation import CardRepresentation
from action_representation import ActionRepresentation
from siamese_net import PseudoSiameseNet, logits_to_probs, clone_model_weights


def build_card_rep_for_state(state: str) -> CardRepresentation:
    """
    Given a 'state' like 'Preflop', 'Flop', 'Turn', 'River', or 'Showdown',
    build an example CardRepresentation. This is a toy demonstration:
      - Preflop: sets hole cards only
      - Flop: sets hole + flop
      - Turn: sets hole + flop + turn
      - River or Showdown: sets hole + flop + turn + river
    """
    cr = CardRepresentation()

    # For demonstration, we always use the same hole cards: As, Ac
    cr.set_preflop([(12,3), (12,2)])  # rank=12 => Ace, suits=3 => clubs, 2 => diamonds

    if state in ['Flop', 'Turn', 'River', 'Showdown']:
        flop_cards = [(7,1), (3,3), (9,2)]  # e.g. 8d, 4s, Tc
        cr.set_flop(flop_cards)

    if state in ['Turn', 'River', 'Showdown']:
        turn_card = (5,0)  # 6h
        cr.set_turn(turn_card)

    if state in ['River', 'Showdown']:
        river_card = (11,3) # Qs
        cr.set_river(river_card)

    return cr


def build_action_rep_for_state(action_history: str, client_pos: int) -> ActionRepresentation:
    """
    Builds an ActionRepresentation from the complete betting history so far.
    
    Args:
        action_history (str): A string of actions (e.g., "b200c/kk/b300") representing the full betting history.
        client_pos (int): The hero’s seat (0 for BB, 1 for SB). Used to determine who acts first in preflop.
        
    Returns:
        ActionRepresentation: An object whose action_tensor reflects all actions taken up to the current state.
    """
    from action_representation import ActionRepresentation
    # Instantiate the full action representation object
    ar = ActionRepresentation(nb=9, max_actions_per_round=6, rounds=4)
    
    # Split the action history by rounds (using '/' as delimiter)
    rounds_list = action_history.split('/')
    # Initialize the pot as the sum of the blinds (for heads-up, BB + SB)
    pot = BIG_BLIND + SMALL_BLIND  # e.g. 100 + 50 = 150

    # Process each round in sequence.
    for round_id, round_actions in enumerate(rounds_list):
        # Determine starting player for the round:
        # Preflop (round_id==0): small blind (seat 1) acts first;
        # Postflop: big blind (seat 0) acts first.
        current_player = 1 if round_id == 0 else 0
        action_index_in_round = 0
        i = 0
        while i < len(round_actions):
            c = round_actions[i]
            # For simple actions (fold, check, call), use a fixed mapping.
            if c in ['f', 'k', 'c']:
                mapping = {'f': 0, 'k': 1, 'c': 2}
                action_idx = mapping[c]
                i += 1
            # For bet actions: note that raises are also marked with 'b'
            elif c == 'b':
                j = i + 1
                while j < len(round_actions) and round_actions[j].isdigit():
                    j += 1
                # Extract the numeric bet portion (e.g., for "b300" extract "300")
                bet_str = round_actions[i+1:j]
                bet_amount = int(bet_str)
                i = j
                # For preflop, subtract the blind already contributed.
                if round_id == 0:
                    if current_player == 1:
                        effective_bet = bet_amount - SMALL_BLIND
                    else:
                        effective_bet = bet_amount - BIG_BLIND
                else:
                    effective_bet = bet_amount
                # Compute the ratio relative to the current pot.
                ratio_val = effective_bet / pot if pot > 0 else 0
                # Map the ratio to a discrete action index:
                if ratio_val < 0.6:
                    action_idx = 3
                elif ratio_val < 0.9:
                    action_idx = 4
                elif ratio_val < 1.2:
                    action_idx = 5
                elif ratio_val < 1.7:
                    action_idx = 6
                elif ratio_val < 2.2:
                    action_idx = 7
                else:
                    action_idx = 8
                # Update the pot with the effective bet.
                pot += effective_bet
            else:
                # Skip any unexpected character.
                i += 1
                continue
            # Add the action into the representation.
            try:
                ar.add_action(round_id, action_index_in_round, current_player, action_idx)
            except Exception as e:
                print(f"Error adding action in round {round_id}: {e}")
            action_index_in_round += 1
            # Alternate the acting player.
            current_player = 1 - current_player
    return ar








def run_one_iteration(iter_idx: int, old_policy_net: PseudoSiameseNet, new_policy_net: PseudoSiameseNet,
                      optimizer: optim.Optimizer):
    """
    Demonstrates a single iteration (episode) with states=[Preflop,Flop,Turn,River,Showdown]
    and rewards=[-20, -20, -80, 0, 240].
    We'll do partial PPO logic: 
      - compute advantage,
      - old/new policy ratio,
      - trinal-clip policy loss,
      - a toy value loss, 
      - gradient update => new_policy_net changes.
    """
    print(f"\n=== Iteration {iter_idx} ===")
    states = ['Preflop','Flop','Turn','River','Showdown']
    rewards = [-20, -20, -80, 0, 240]

    # We'll track some metrics
    total_pol_loss = 0
    total_val_loss = 0
    steps_count = 0

    # Build a model-based value function using the *new* net 
    # (In typical PPO, the value function is updated simultaneously with the new policy.)
    model_value_func = make_model_value_function(new_policy_net, build_card_rep_for_state, build_action_rep_for_state)

    for i, st in enumerate(states[:-1]):  # skip Showdown itself
        # 1) Compute advantage
        future_rewards = rewards[i:]
        future_states  = states[i:]
        advantage = a_gae(future_rewards, future_states, model_value_func, gamma=0.999, lambda_=0.99)

        # 2) Build card/action reps for this state
        card_rep = build_card_rep_for_state(st)
        action_rep = build_action_rep_for_state(st)
        action_t, card_t = to_torch_input(card_rep, action_rep)

        # 3) old policy => old_probs
        with torch.no_grad():
            old_logits, _ = old_policy_net(action_t, card_t)
            old_probs = logits_to_probs(old_logits)[0].cpu().numpy()

        # 4) new policy => new_probs + new_value
        new_logits, new_value = new_policy_net(action_t, card_t)
        new_probs_t = logits_to_probs(new_logits)[0]
        new_probs = new_probs_t.detach().cpu().numpy()

        # 5) sample an action from the *new* policy
        action_idx = np.random.choice(len(new_probs), p=new_probs)

        # 6) ratio = new_probs[action_idx]/old_probs[action_idx]
        ratio_val = ratio(old_probs, new_probs, action_idx)

        # 7) policy loss
        deltas = get_deltas(st)
        pol_loss_val = tc_loss_function(ratio_val, advantage, epsilon=0.2, deltas=deltas)

        # 8) value loss
        #    compute r_gamma from future rewards
        r_g = r_gamma(np.array(future_rewards), gamma=0.999)
        val_loss_val = v_loss(r_g, st, deltas, value_state=model_value_func)

        # 9) build a toy combined loss => do a gradient update
        #   - incorporate pol_loss_val and val_loss_val into the PyTorch graph 
        #   - We'll do a negative log(prob_of_action) scaled by pol_loss_val,
        #     plus MSE( new_value, val_loss_val ).
        chosen_log_prob = torch.log(new_probs_t[action_idx] + 1e-8)
        pol_loss_t = torch.tensor(pol_loss_val, dtype=torch.float32)
        val_loss_t = torch.tensor(val_loss_val, dtype=torch.float32)

        combined_loss = - pol_loss_t * chosen_log_prob + (new_value[0] - val_loss_t)**2

        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        total_pol_loss += pol_loss_val
        total_val_loss += val_loss_val
        steps_count += 1

        print(f"  State={st}, ratio={ratio_val:.3f}, advantage={advantage:.2f}, action_idx={action_idx}")
        print(f"    pol_loss={pol_loss_val:.3f}, val_loss={val_loss_val:.3f}")

    if steps_count > 0:
        print(f"=> iteration {iter_idx} done. avg pol_loss={total_pol_loss/steps_count:.3f}, avg val_loss={total_val_loss/steps_count:.3f}")


def main():
    # 1) Create two siamese nets: old policy & new policy
    old_policy_net = PseudoSiameseNet()
    new_policy_net = PseudoSiameseNet()

    # Initially clone new -> old so they start the same
    clone_model_weights(new_policy_net, old_policy_net)

    optimizer = optim.Adam(new_policy_net.parameters(), lr=1e-3)

    # Iteration 1
    run_one_iteration(1, old_policy_net, new_policy_net, optimizer)

    # Now new_policy_net changed => copy it to old for the next iteration
    clone_model_weights(new_policy_net, old_policy_net)

    # Iteration 2
    run_one_iteration(2, old_policy_net, new_policy_net, optimizer)

    print("\nDone. You can add more iterations or logic as needed.")


if __name__ == "__main__":
    main()
