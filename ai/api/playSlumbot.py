import requests
import sys
import argparse
import random
import numpy as np
import urllib3
import sys
import os

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

def ChooseActionAI(parsed_state, hole_cards, board, client_pos):
    """
    Simple logic:
      - If it's not our turn => return ''
      - If there's some bet (to_call>0), we call or fold
      - If no bet to call => random check or bet
      - If seat=1 preflop => we might clamp a minimum open, etc. 
    """
    if parsed_state['hand_over']:
        return ''

    seat = parsed_state['next_to_act']
    if seat == -1:
        return ''
    if seat != client_pos:
        return ''

    hero_st = parsed_state['hero_street']
    vill_st = parsed_state['vill_street']

    print(f"Hero street={hero_st}, Villain street={vill_st}")

    # how many more do we owe if we want to continue?
    to_call = vill_st - hero_st

    pot_so_far = parsed_state['pot_total'] + hero_st + vill_st
    my_stack = 20000

    if to_call > 0:
        # can't check if we owe chips => must call or fold
        if to_call > my_stack * 0.4:
            return 'f'  # large portion of stack => fold
        else:
            return 'c'  # call
    else:
        # no bet => random check or bet
        if random.random() < 0.5:
            return 'k'
        else:
            # bet half pot
            half_pot = pot_so_far // 2
            new_total = hero_st + half_pot
            if new_total > my_stack:
                new_total = my_stack

            # (Optional) if seat=1 preflop => enforce min open=200
            if parsed_state['street'] == 0 and seat == 1:
                if new_total < 200:
                    new_total = 200

            return f"b{new_total}"



def PlayHand(token):
    resp = NewHand(token)
    if 'token' in resp:
        token = resp['token']

    action_str = resp.get('action','')
    client_pos = resp.get('client_pos', 0)
    hole_cards = resp.get('hole_cards', [])
    board      = resp.get('board', [])
    winnings   = resp.get('winnings', None)

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
            return token, winnings

        parsed = parse_action_enhanced(action_str, client_pos)
        if parsed['error']:
            print("Error:", parsed['error'])
            print("====================================\n")
            return token, 0

        if parsed['hand_over']:
            final_win = resp.get('winnings', 0)
            print(f"Hand Over => pot={parsed['pot_total']}, winnings={final_win}")
            print("====================================\n")
            return token, final_win

        if 'winnings' in resp and resp['winnings'] is not None:
            w = resp['winnings']
            print(f"Slumbot indicates hand ended => w={w}")
            print("====================================\n")
            return token, w

        seat_to_act = parsed['next_to_act']
        street_name = get_street_name(parsed['street'])
        print(f"\nStreet: {street_name}, next_to_act={seat_to_act}, action so far='{action_str}'")

        ai_move = ChooseActionAI(parsed, hole_cards, board, client_pos)
        if not ai_move:
            print("No action from hero => presumably Slumbot's turn or hand ended.")
            print("Exiting. The hand might continue from Slumbot's perspective.")
            print("====================================\n")
            return token, 0

        print(f"Hero's action => '{ai_move}'")
        resp = Act(token, ai_move)
        new_token = resp.get('token', token)
        if new_token:
            token = new_token

        action_str = resp.get('action','')
        board      = resp.get('board', board)
        winnings   = resp.get('winnings', None)
        print(f"Updated action => '{action_str}', Board={board if board else 'No board'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    parser.add_argument('--num_hands', type=int, default=5)
    args = parser.parse_args()

    token = None
    if args.username and args.password:
        token = Login(args.username, args.password)

    total_winnings = 0
    for h in range(args.num_hands):
        print(f"\n=== Playing hand #{h+1} ===")
        token, w = PlayHand(token)
        total_winnings += (w or 0)

    print(f"\nDONE. total_winnings={total_winnings}")

if __name__ == '__main__':
    main()