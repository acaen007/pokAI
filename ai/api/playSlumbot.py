import requests
import sys
import argparse
import random
import numpy as np
import urllib3
import sys
import os

from replay import parse_card
from experiment import build_action_rep_for_state, to_torch_input
from siamese_net import logits_to_probs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
        print('Status code:', response.status_code)
        try:
            print('Error response:', response.json())
        except ValueError:
            pass
        sys.exit(-1)

    r = response.json()
    if 'error_msg' in r:
        print('Error:', r['error_msg'])
        sys.exit(-1)

    token = r.get('token')
    if not token:
        print('Did not get token in response to /api/login')
        sys.exit(-1)
    return token


def NewHand(token):
    data = {}
    if token:
        data['token'] = token
    response = requests.post(f'https://{host}/api/new_hand', json=data, verify=False)
    if response.status_code != 200:
        print('Status code:', response.status_code)
        try:
            print('Error response:', response.json())
        except ValueError:
            pass
        sys.exit(-1)

    r = response.json()
    if 'error_msg' in r:
        print('Error:', r['error_msg'])
        sys.exit(-1)
    return r


def Act(token, action):
    data = {'token': token, 'incr': action}
    response = requests.post(f'https://{host}/api/act', json=data, verify=False)
    if response.status_code != 200:
        print('Status code:', response.status_code)
        try:
            print('Error response:', response.json())
        except ValueError:
            pass
        sys.exit(-1)

    r = response.json()
    if 'error_msg' in r:
        print('Error:', r['error_msg'])
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
    # A simple mapping example based on our earlier pot-fraction buckets:
    # 0 => fold ('f'), 1 => check ('k'), 2 => call ('c'),
    # 3 => small bet, 4 => medium bet, 5 => pot bet, etc.
    hero_st = parsed_state['hero_street']
    vill_st = parsed_state['vill_street']
    to_call = vill_st - hero_st
    pot_so_far = parsed_state['pot_total'] + hero_st + vill_st
    if action_idx == 0:
        return 'f'
    elif action_idx == 1:
        return 'k'
    elif action_idx == 2:
        return 'c'
    elif action_idx == 8:
        return f"b{my_stack}"
    else:
        # For bet actions, we simply choose a bet based on a fraction.
        # Here we set a default fraction based on the index.
        fractions = {3: 0.5, 4: 0.75, 5: 1.0, 6: 1.5, 7: 2.0}
        frac = fractions.get(action_idx, 0.5)
        half_pot = int(pot_so_far * frac)
        new_total = hero_st + half_pot
        if new_total > my_stack:
            new_total = my_stack
        return f"b{new_total}"

def ChooseActionAI(parsed_state, hole_cards, board, client_pos, policy_net=None):
    """
    Determines our action.
      - If policy_net is provided, builds state representations from the current betting history
        and uses the model’s outputs to sample an action.
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
        # Build the card representation.
        from card_representation import CardRepresentation
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
        # Build the action representation from the complete betting history.
        # (action_str holds the full history up to now)
        action_rep = build_action_rep_for_state(parsed_state.get('action_str', ""), client_pos)
        # Convert to torch tensors.
        action_t, card_t = to_torch_input(card_rep.card_tensor, action_rep.action_tensor)
        logits, _ = policy_net.forward(action_t, card_t)
        probs = logits_to_probs(logits)[0].detach().cpu().numpy()
        action_idx = np.random.choice(len(probs), p=probs)

        print("Playing action based on policy network.")
        print(f"Action probs: {probs}")
        return index_to_action_string(action_idx, parsed_state, my_stack)
    else:
        print("Falling back to heuristic logic.")
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




def PlayHand(token, policy_net=None):
    resp = NewHand(token)
    if 'token' in resp:
        token = resp['token']

    action_str = resp.get('action', '')
    client_pos = resp.get('client_pos', 0)
    hole_cards = resp.get('hole_cards', [])
    board = resp.get('board', [])
    winnings = resp.get('winnings', None)

    print("\n====================================")
    print("NEW HAND STARTED")
    print(f"Token: {token}")
    print(f"Hero seat={client_pos} => {'Big Blind' if client_pos==0 else 'Small Blind'}")
    print(f"Hole cards: {hole_cards}, Board: {board if board else 'No board'}")
    print(f"Initial action: '{action_str}'")

    while True:
        if winnings is not None:
            print(f"Hand ended immediately, winnings={winnings}")
            print("====================================\n")
            final_action = action_str
            return token, winnings, final_action, hole_cards, board, client_pos

        parsed = parse_action_enhanced(action_str, client_pos)
        if parsed['error']:
            print("Error:", parsed['error'])
            print("====================================\n")
            final_action = action_str
            return token, 0, final_action, hole_cards, board, client_pos

        if parsed['hand_over']:
            final_win = resp.get('winnings', 0)
            print(f"Hand Over => pot={parsed['pot_total']}, winnings={final_win}")
            print("====================================\n")
            final_action = action_str
            return token, final_win, final_action, hole_cards, board, client_pos

        if 'winnings' in resp and resp['winnings'] is not None:
            w = resp['winnings']
            print(f"Slumbot indicates hand ended => w={w}")
            print("====================================\n")
            final_action = action_str
            return token, w, final_action, hole_cards, board, client_pos

        seat_to_act = parsed['next_to_act']
        street_name = get_street_name(parsed['street'])
        print(f"\nStreet: {street_name}, next_to_act={seat_to_act}, action so far='{action_str}'")

        print(f"Hero's hole cards: {hole_cards}, Board: {board if board else 'No board'}")
        ai_move = ChooseActionAI(parsed, hole_cards, board, client_pos, policy_net)
        if not ai_move:
            print("No action from hero => presumably Slumbot's turn or hand ended.")
            print("Exiting. The hand might continue from Slumbot's perspective.")
            print("====================================\n")
            final_action = action_str
            return token, 0, final_action, hole_cards, board, client_pos

        print(f"Hero's action => '{ai_move}'")
        resp = Act(token, ai_move)
        new_token = resp.get('token', token)
        if new_token:
            token = new_token

        action_str = resp.get('action', '')
        board = resp.get('board', board)
        winnings = resp.get('winnings', None)
        print(f"Updated action => '{action_str}', Board={board if board else 'No board'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    parser.add_argument('--num_hands', type=int, default=50)
    args = parser.parse_args()

    token = None
    if args.username and args.password:
        token = Login(args.username, args.password)

    # Create or load your trained policy network.
    # For example, here we create a new instance (replace with your trained model as needed):
    from siamese_net import PseudoSiameseNet
    policy_net = PseudoSiameseNet()

    total_winnings = 0
    for h in range(args.num_hands):
        print(f"\n=== Playing hand #{h+1} ===")
        token, w, final_action, hole_cards, board, client_pos = PlayHand(token, policy_net=policy_net)
        total_winnings += (w or 0)

        # Append replay info to a text file (one line per hand):
        with open("replay.txt", "a") as replay_file:
            replay_file.write(f"{h+1},{final_action},{board},{hole_cards},{client_pos},{w}\n")

    print(f"\nDONE. total_winnings={total_winnings}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--username', type=str)
#     parser.add_argument('--password', type=str)
#     parser.add_argument('--num_hands', type=int, default=50)
#     args = parser.parse_args()

#     token = None
#     if args.username and args.password:
#         token = Login(args.username, args.password)

#     total_winnings = 0
#     for h in range(args.num_hands):
#         print(f"\n=== Playing hand #{h+1} ===")
#         token, w, final_action, hole_cards, board, client_pos = PlayHand(token, policy_net=None)
#         total_winnings += (w or 0)
    
#         # Append replay info to a text file (one line per hand):
#         with open("replay.txt", "a") as replay_file:
#             # Format: index, final action string, hole cards, board, client_pos, winnings
#             replay_file.write(f"{h+1},{final_action},{board},{hole_cards},{client_pos},{w}\n")

#     print(f"\nDONE. total_winnings={total_winnings}")

if __name__ == '__main__':
    main()