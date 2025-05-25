import requests
import sys
import argparse
import random
import numpy as np
import torch
import urllib3
import sys
import os

from .replay import parse_card
from experiment import build_action_rep_for_state
from siamese_net import logits_to_probs, to_torch_input
from card_representation import CardRepresentation

from debug_utils import debug_print
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

# ------------------------------
# SLUMBOT CONSTANTS
# ------------------------------
host = 'slumbot.com'
NUM_STREETS = 4
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000


# =============================================================================
# 1) SLUMBOT API
# =============================================================================
def Login(username, password):
    data = {"username": username, "password": password}
    response = requests.post(f'https://{host}/api/login', json=data, verify=False)
    if response.status_code != 200:
        debug_print('Status code:', response.status_code)
        try:
            debug_print('Error response:', response.json())
        except ValueError:
            pass
        sys.exit(-1)

    r = response.json()
    if 'error_msg' in r:
        debug_print('Error:', r['error_msg'])
        sys.exit(-1)

    token = r.get('token')
    if not token:
        debug_print('Did not get token in response to /api/login')
        sys.exit(-1)
    return token


def NewHand(token):
    data = {}
    if token:
        data['token'] = token
    response = requests.post(f'https://{host}/api/new_hand', json=data, verify=False)
    if response.status_code != 200:
        debug_print('Status code:', response.status_code)
        try:
            debug_print('Error response:', response.json())
        except ValueError:
            pass
        sys.exit(-1)

    r = response.json()
    if 'error_msg' in r:
        debug_print('Error:', r['error_msg'])
        sys.exit(-1)
    return r


def Act(token, action):
    data = {'token': token, 'incr': action}
    response = requests.post(f'https://{host}/api/act', json=data, verify=False)
    if response.status_code != 200:
        debug_print('Status code:', response.status_code)
        try:
            debug_print('Error response:', response.json())
        except ValueError:
            pass
        sys.exit(-1)

    r = response.json()
    if 'error_msg' in r:
        debug_print('Error:', r['error_msg'])
        sys.exit(-1)
    return r


# =============================================================================
# 2) HELPER: street name
# =============================================================================
def get_street_name(st):
    if st == 0:
        return "Preflop"
    elif st == 1:
        return "Flop"
    elif st == 2:
        return "Turn"
    elif st == 3:
        return "River"
    return "Showdown"


# =============================================================================
# 3) ENFORCE SB MIN RAISE=200 PREFLOP
# =============================================================================
def parse_action_enhanced(action_str: str, client_pos: int):
    """
    HEADS-UP ONLY. seat=0 => posted 100, seat=1 => posted 50, seat=1 acts first preflop, seat=0 acts first postflop.
    We do NOT raise an error for small raises here. The AI code ensures it won't produce them from seat=1 preflop.
    """
    st = 0
    pot_total = 0

    hero_street = (BIG_BLIND if client_pos == 0 else SMALL_BLIND)
    vill_street = (SMALL_BLIND  if client_pos == 0 else BIG_BLIND)

    # seat=1 next preflop
    pos = 1

    error = None
    isHandOver = False
    i = 0

    while i < len(action_str):
        if st >= NUM_STREETS:
            error = f"Exceeded # streets in action: {action_str}"
            break

        c = action_str[i]
        i += 1

        # how many to call
        if client_pos == 0:
            # hero is seat=0
            if pos == 0:
                to_call = vill_street - hero_street
            else:
                to_call = hero_street - vill_street
        else:
            # hero seat=1
            if pos == 1:
                to_call = vill_street - hero_street
            else:
                to_call = hero_street - vill_street

        if c == '/':
            pot_total += (hero_street + vill_street)
            hero_street = 0
            vill_street = 0
            st += 1
            if st >= NUM_STREETS:
                pos = -1
                isHandOver = True
                break
            pos = 0  # postflop seat=0 first
            continue
        elif c == 'k':
            if to_call > 0:
                error = f"Illegal check: needed to call {to_call}"
                break
            pos = 1 - pos
        elif c == 'c':
            if to_call <= 0:
                error = "Illegal call: no bet to call"
                break
            if pos == client_pos:
                hero_street += to_call
            else:
                vill_street += to_call
            pos = 1 - pos
        elif c == 'f':
            pos = -1
            isHandOver = True
            break
        elif c == 'b':
            j = i
            while i < len(action_str) and action_str[i].isdigit():
                i += 1
            if i == j:
                error = "Missing bet size after 'b'"
                break
            try:
                new_this_street = int(action_str[j:i])
            except ValueError:
                error = "Bet size not integer"
                break

            if pos == client_pos:
                hero_street = new_this_street
            else:
                vill_street = new_this_street
            pos = 1 - pos
        else:
            error = f"Unexpected char {c}"
            break

    if not error and (isHandOver or pos == -1):
        pot_total += (hero_street + vill_street)

    return {
        'street': st,
        'pot_total': pot_total,
        'hero_street': hero_street,
        'vill_street': vill_street,
        'next_to_act': pos,
        'error': error,
        'hand_over': isHandOver or (pos == -1)
    }

# ------------------------------
# MODIFY CHOOSEACTIONAI TO USE POLICY OUTPUTS
# ------------------------------
# (Assumes a global policy_net variable is set after training.)
# Also note: index_to_action_string() is a helper that maps a chosen action index (0-8)
# to the appropriate API action string given the parsed state. Adjust its logic as needed.
def index_to_action_string(action_idx, parsed_state, my_stack):
    """
    Converts a discrete action index into an API action string,
    ensuring that the bet does not exceed the player's remaining stack.
    
    We assume:
      - parsed_state['hero_street']: contribution in the current street.
      - parsed_state['pot_total']: total chips contributed in previous streets.
    
    We estimate the hero's total contribution as:
         hero_total = hero_street + (pot_total / 2)
    so that the remaining chips = my_stack - hero_total.
    
    The mapping is:
      0 -> fold ('f')
      1 -> check ('k')
      2 -> call ('c')
      3-7 -> various bet sizes (fractions of the current pot)
      8 -> all-in (bet all remaining chips)
    """
    hero_st = parsed_state['hero_street']
    vill_st = parsed_state['vill_street']
    to_call = vill_st - hero_st
    pot_so_far = parsed_state['pot_total'] + hero_st + vill_st

    # Estimate hero's total contribution:
    hero_total = int(hero_st + (parsed_state['pot_total'] / 2))
    remaining = my_stack - hero_total

    if action_idx == 0:
        return 'f'
    elif action_idx == 1:
        return 'k'
    elif action_idx == 2:
        return 'c'
    elif action_idx == 8:
        # All-in: bet all remaining chips in addition to what has already been bet this street.
        return f"b{hero_st + remaining}"
    else:
        # Map indices 3–7 to fractions of the pot.
        fractions = {3: 0.5, 4: 0.75, 5: 1.0, 6: 1.5, 7: 2.0}
        frac = fractions.get(action_idx, 0.5)
        # Compute bet amount as a fraction of the pot.
        bet_amount = int(pot_so_far * frac)
        new_total = hero_st + bet_amount
        # If responding to a bet, enforce a minimum raise of at least one big blind above the opponent.
        if to_call > 0:
            min_raise_total = vill_st + BIG_BLIND
            if new_total < min_raise_total:
                new_total = min_raise_total
        # Ensure we do not exceed the remaining chips.
        if new_total > hero_st + remaining:
            new_total = hero_st + remaining
        return f"b{new_total}"



def compute_legal_mask(parsed_state, my_stack):
    """
    Computes a legal mask vector (length 9) for discrete actions based on the parsed state and available stack.
    
    The parsed_state should contain:
      - 'hero_street': the hero's contribution in the current betting round.
      - 'vill_street': the opponent's contribution in the current betting round.
      - 'pot_total': the total chips contributed in previous rounds.
      - 'street': the current round (0 for Preflop, 1 for Flop, etc.)
    
    We estimate the total contributions as:
         hero_total = hero_street + (pot_total / 2)
         vill_total = vill_street + (pot_total / 2)
    so that remaining chips for hero = my_stack - hero_total.
    
    Additional rules:
      - **If the opponent is all-in:**  
            If vill_total >= my_stack, then only call (action index 2) and fold (action index 0) are legal.
      - **If facing an outstanding bet (to_call > 0):**
          * Allow call (2) always.
          * Allow fold (0) only when appropriate (e.g. preflop or if hero has already acted).
          * For raise options (indices 3–7):
                - If hero is acting first in a round beyond preflop (hero_street == 0 and street > 0),
                  the new total bet must be at least 2×vill_street.
                - If hero has already acted (hero_street > 0), then the new total must be at least
                  vill_street + (vill_street - hero_street) (i.e. at least as large as the last raise).
      - **If no bet is outstanding (to_call <= 0):**
          Folding is disallowed (only check, bet, or all‑in are legal).
      - **All-in option (index 8):**
          Allowed if chips remain, except when the opponent is all-in.
    """
    import numpy as np
    legal = np.zeros(9, dtype=np.float32)
    
    hero_st = parsed_state['hero_street']
    vill_st = parsed_state['vill_street']
    to_call = vill_st - hero_st
    pot_so_far = parsed_state['pot_total'] + hero_st + vill_st
    current_round = parsed_state.get('street', 0)
    
    # Estimate total chips already contributed.
    hero_total = hero_st + (parsed_state['pot_total'] / 2)
    vill_total = vill_st + (parsed_state['pot_total'] / 2)
    remaining = my_stack - hero_total
    
    # If the opponent is already all-in, only call or fold are allowed.
    if vill_total >= my_stack:
        legal[0] = 1  # fold
        legal[2] = 1  # call (which would go all-in)
        return legal
    
    # Define discrete raise options (indices 3 to 7) with corresponding pot fractions.
    fractions = {3: 0.5, 4: 0.75, 5: 1.0, 6: 1.5, 7: 2.0}
    
    if to_call > 0:
        # Facing an outstanding bet.
        legal[2] = 1  # call is always legal.
        # Allow fold only if it makes sense (e.g. preflop or if hero has already acted).
        if current_round == 0 or hero_st > 0:
            legal[0] = 1
        
        # Determine minimum total bet required:
        if hero_st == 0 and current_round > 0:
            # When hero is first to act beyond preflop, minimum total bet is 2×vill_st.
            min_total = 2 * vill_st
        elif hero_st > 0:
            # When hero has already acted, minimum total = vill_st + (vill_st - hero_st)
            min_total = vill_st + (vill_st - hero_st)
        else:
            # Preflop: use villain's contribution as minimum.
            min_total = vill_st
        
        # For each raise option, check if the new total meets the minimum and fits within remaining chips.
        for idx, frac in fractions.items():
            bet_amount = int(pot_so_far * frac)  # additional bet amount for this option.
            new_total = hero_st + bet_amount       # new total contribution if chosen.
            if new_total >= min_total and bet_amount <= remaining:
                legal[idx] = 1
        
        # All-in (index 8) is legal if there are chips remaining.
        if remaining > 0:
            legal[8] = 1
    else:
        # No outstanding bet (acting first or after a check).
        # Folding is disallowed.
        legal[1] = 1  # check is legal.
        for idx, frac in fractions.items():
            bet_amount = int(pot_so_far * frac)
            new_total = hero_st + bet_amount
            if new_total > hero_st and bet_amount <= remaining:
                legal[idx] = 1
        if remaining > 0:
            legal[8] = 1
            
    return legal





def ChooseActionAI(parsed_state, hole_cards, board, client_pos, policy_net=None, device=None):
    """
    Determines our action.
      - If policy_net is provided, builds the state representations (card and action)
        from the current betting history and uses the model’s outputs to sample an action.
      - The network's output is masked so that only legal actions (per computed mask) get probability mass.
      - Otherwise, falls back to heuristic logic.
    """
    if parsed_state['hand_over']:
        return ''
    
    seat = parsed_state['next_to_act']
    if seat != client_pos:
        return ''
    
    hero_st = parsed_state['hero_street']
    vill_st = parsed_state['vill_street']
    to_call = vill_st - hero_st
    pot_so_far = parsed_state['pot_total'] + hero_st + vill_st
    my_stack = STACK_SIZE

    if policy_net is not None:
        # Build card representation from hole cards and board.
        card_rep = CardRepresentation()
        parsed_hole = [parse_card(c) for c in hole_cards]
        card_rep.set_preflop(parsed_hole)
        street = parsed_state['street']
        if street >= 1 and len(board) >= 3:
            flop = [parse_card(c) for c in board[:3]]
            card_rep.set_flop(flop)
        if street >= 2 and len(board) >= 4:
            card_rep.set_turn(parse_card(board[3]))
        if street >= 3 and len(board) >= 5:
            card_rep.set_river(parse_card(board[4]))
        
        # Build the action representation from the betting history.
        # (We assume parsed_state has the full action string stored as 'action_str'.)
        action_rep = build_action_rep_for_state(parsed_state.get('action_str', ""), client_pos)
        
        # Convert representations to torch tensors.
        action_t, card_t = to_torch_input(card_rep.card_tensor, action_rep.action_tensor, device=device)
        
        # Run through the policy network.
        logits, _ = policy_net.forward(action_t, card_t)  # logits shape: [1, nb]
        logits = logits[0]  # remove batch dimension
        
        # Compute legal mask.
        legal_mask = compute_legal_mask(parsed_state, my_stack)  # numpy vector of shape (9,)
        legal_mask_tensor = torch.tensor(legal_mask, dtype=torch.float32, device=logits.device)
        
        # Mask logits: set illegal actions to a large negative number.
        masked_logits = logits + (1 - legal_mask_tensor) * (-1e10)
        # Compute probabilities over legal actions.
        masked_probs = torch.softmax(masked_logits, dim=0)
        masked_probs_np = masked_probs.detach().cpu().numpy()
        
        # (Optional debug: print the masked probabilities)
        debug_print("Playing action with policy network.")
        debug_print("Action probs:", masked_probs_np)
        
        # Sample an action from the masked probabilities.
        action_idx = np.random.choice(len(masked_probs_np), p=masked_probs_np)
        # Get the API action string from our mapping.
        ai_action = index_to_action_string(action_idx, parsed_state, my_stack)
        return ai_action
    else:
        debug_print("No policy network provided. Using heuristic logic.")
        # Fallback heuristic.
        if to_call > 0:
            return 'c' if to_call <= my_stack * 0.4 else 'f'
        else:
            if random.random() < 0.5:
                return 'k'
            else:
                half_pot = pot_so_far // 2
                new_total = hero_st + half_pot
                if new_total > my_stack:
                    new_total = my_stack
                if parsed_state['street'] == 0 and seat == 1 and new_total < 200:
                    new_total = 200
                return f"b{new_total}"


def PlayHand(token, policy_net=None, device=None):
    resp = NewHand(token)
    if 'token' in resp:
        token = resp['token']

    action_str = resp.get('action', '')
    client_pos = resp.get('client_pos', 0)
    hole_cards = resp.get('hole_cards', [])
    board = resp.get('board', [])
    winnings = resp.get('winnings', None)

    debug_print("\n====================================")
    debug_print("NEW HAND STARTED")
    debug_print(f"Token: {token}")
    debug_print(f"Hero seat={client_pos} => {'Big Blind' if client_pos==0 else 'Small Blind'}")
    debug_print(f"Hole cards: {hole_cards}, Board: {board if board else 'No board'}")
    debug_print(f"Initial action: '{action_str}'")

    while True:
        if winnings is not None:
            debug_print(f"Hand ended immediately, winnings={winnings}")
            debug_print("====================================\n")
            final_action = action_str
            return token, winnings, final_action, hole_cards, board, client_pos

        parsed = parse_action_enhanced(action_str, client_pos)
        if parsed['error']:
            debug_print("Error:", parsed['error'])
            debug_print("====================================\n")
            final_action = action_str
            return token, 0, final_action, hole_cards, board, client_pos

        if parsed['hand_over']:
            final_win = resp.get('winnings', 0)
            debug_print(f"Hand Over => pot={parsed['pot_total']}, winnings={final_win}")
            debug_print("====================================\n")
            final_action = action_str
            return token, final_win, final_action, hole_cards, board, client_pos

        if 'winnings' in resp and resp['winnings'] is not None:
            w = resp['winnings']
            debug_print(f"Slumbot indicates hand ended => w={w}")
            debug_print("====================================\n")
            final_action = action_str
            return token, w, final_action, hole_cards, board, client_pos

        seat_to_act = parsed['next_to_act']
        street_name = get_street_name(parsed['street'])
        debug_print(f"\nStreet: {street_name}, next_to_act={seat_to_act}, action so far='{action_str}'")

        debug_print(f"Hero's hole cards: {hole_cards}, Board: {board if board else 'No board'}")
        ai_move = ChooseActionAI(parsed, hole_cards, board, client_pos, policy_net, device=device)
        if not ai_move:
            debug_print("No action from hero => presumably Slumbot's turn or hand ended.")
            debug_print("Exiting. The hand might continue from Slumbot's perspective.")
            debug_print("====================================\n")
            final_action = action_str
            return token, 0, final_action, hole_cards, board, client_pos

        debug_print(f"Hero's action => '{ai_move}'")
        resp = Act(token, ai_move)
        new_token = resp.get('token', token)
        if new_token:
            token = new_token

        action_str = resp.get('action', '')
        board = resp.get('board', board)
        winnings = resp.get('winnings', None)
        debug_print(f"Updated action => '{action_str}', Board={board if board else 'No board'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    parser.add_argument('--num_hands', type=int, default=200)
    args = parser.parse_args()

    debug_print("=========================================")
    debug_print(args.username, args.password)

    token = None
    if args.username and args.password:
        token = Login(args.username, args.password)

    # Create or load your trained policy network.
    # For example, here we create a new instance (replace with your trained model as needed):
    from siamese_net import PseudoSiameseNet
    policy_net = PseudoSiameseNet()

    total_winnings = 0
    for h in range(args.num_hands):
        debug_print(f"\n=== Playing hand #{h+1} ===")
        token, w, final_action, hole_cards, board, client_pos = PlayHand(token, policy_net=policy_net)
        total_winnings += (w or 0)

        # Append replay info to a text file (one line per hand):
        with open("ai/replay.txt", "a") as replay_file:
            replay_file.write(f"{h+1},{final_action},{board},{hole_cards},{client_pos},{w}\n")

    debug_print(f"\nDONE. total_winnings={total_winnings}")
    debug_print("mBB/hand:", total_winnings / (args.num_hands * BIG_BLIND))

    

if __name__ == '__main__':
    main()