"""
File: ai_main.py

Demonstrates a robust Slumbot integration:
 - Logging in (optional)
 - Repeatedly starting new hands
 - Parsing Slumbot's action string with parse_action_enhanced
 - Deciding our action with ChooseActionAI
 - Sending that action to Slumbot via Act()
"""

import requests
import sys
import argparse
import random
import numpy as np
import urllib3

# ------------------------------
# DISABLE SSL WARNINGS (Optional)
# ------------------------------
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
# 1) SLUMBOT API FUNCTIONS
#    (Login, NewHand, Act), from your provided sample
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
# 2) ROBUST PARSER: parse_action_enhanced
#    Tracks pot sizes, contributions, etc.
# =============================================================================

def get_street_name(st):
    """
    Convert street index (0..3) to a human-readable name.
    0 => Preflop, 1 => Flop, 2 => Turn, 3 => River
    """
    if st == 0:
        return "Preflop"
    elif st == 1:
        return "Flop"
    elif st == 2:
        return "Turn"
    elif st == 3:
        return "River"
    return "Showdown"


def parse_action_enhanced(action_str: str, client_pos: int):
    """
    Parses the entire Slumbot action string, tracking:
      - street (0..3)
      - pot_total (sum of previous streets)
      - hero_street, vill_street: how many chips each side contributed THIS street
      - next_to_act: seat that acts next, or -1 if hand over
      - error: str if illegal
      - hand_over: bool
    """

    st = 0
    pot_total = 0
    # If client_pos=0 => we posted big blind=100, villain posted small blind=50 (preflop).
    # If client_pos=1 => we posted small blind=50, villain posted big blind=100.
    hero_street = BIG_BLIND if client_pos == 0 else SMALL_BLIND
    vill_street = SMALL_BLIND if client_pos == 0 else BIG_BLIND

    # Next to act preflop => seat=1 if client_pos=0, else seat=0
    pos = 1 if client_pos == 0 else 0

    isHandOver = False
    error = None
    i = 0
    while i < len(action_str):
        if st >= NUM_STREETS:
            error = f"Exceeded # streets in action: {action_str}"
            break
        c = action_str[i]
        i += 1

        # how much hero owes to call?
        if client_pos == 0:
            to_call = vill_street - hero_street if pos == 0 else hero_street - vill_street
        else:
            to_call = hero_street - vill_street if pos == 1 else vill_street - hero_street

        if c == '/':
            # street boundary
            pot_total += (hero_street + vill_street)
            hero_street = 0
            vill_street = 0
            st += 1
            if st >= NUM_STREETS:
                pos = -1
                isHandOver = True
                break
            pos = 0 if st > 0 else 1
            continue

        elif c == 'k':
            # check => must owe 0
            if to_call > 0:
                error = f"Illegal check: needed to call {to_call}"
                break
            pos = 1 - pos

        elif c == 'c':
            # call => must owe > 0
            if to_call <= 0:
                error = "Illegal call: no bet to call"
                break
            if pos == client_pos:
                hero_street += to_call
            else:
                vill_street += to_call
            pos = 1 - pos

        elif c == 'f':
            # fold => hand ends
            pos = -1
            isHandOver = True
            break

        elif c == 'b':
            j = i
            while i < len(action_str) and action_str[i].isdigit():
                i += 1
            if i == j:
                error = "Missing bet size"
                break
            try:
                new_this_street = int(action_str[j:i])
            except ValueError:
                error = "Bet size not integer"
                break
            if pos == client_pos:
                bet_size = new_this_street - hero_street
                if bet_size < 0:
                    error = f"Bet size too small {bet_size}"
                    break
                hero_street = new_this_street
            else:
                bet_size = new_this_street - vill_street
                if bet_size < 0:
                    error = f"Bet size too small {bet_size}"
                    break
                vill_street = new_this_street
            pos = 1 - pos
        else:
            error = f"Unexpected char {c}"
            break

    # end while
    if not error:
        if isHandOver or pos == -1:
            # add final partial street
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


# =============================================================================
# 3) AI CODE: ChooseActionAI
# =============================================================================

def ChooseActionAI(parsed_state, hole_cards, board, client_pos):
    """
    A simple (toy) AI that picks an action among ['k','c','f','b###'].
    We interpret the pot size from parsed_state['pot_total'] + each side's street contribution.
    'to_call' is how many more chips we must invest if we want to call.

    Real logic: you'd do your RL or policy net here.
    """
    if parsed_state['hand_over']:
        return ''  # no action, hand ended

    # next_to_act seat
    seat = parsed_state['next_to_act']
    if seat == -1:
        return ''  # hand ended
    if seat != client_pos:
        # Not our turn => return empty
        return ''

    # how many chips have we put in vs opponent this street?
    hero_st = parsed_state['hero_street']
    vill_st = parsed_state['vill_street']
    if client_pos == 0:
        to_call = vill_st - hero_st
    else:
        to_call = hero_st - vill_st

    # base pot
    pot_so_far = parsed_state['pot_total'] + hero_st + vill_st

    # We'll do a random logic approach:
    # If there's something to call, we call if it's not huge, else fold.
    # If no call needed, random check or bet half pot.
    my_stack = 20000  # ignoring detailed track; we assume full stack for example

    if to_call > 0:
        if to_call > my_stack * 0.4:
            # big bet => fold
            return 'f'
        else:
            # call
            return 'c'
    else:
        # no bet => either check or small bet
        if random.random() < 0.5:
            return 'k'
        else:
            # bet half pot: new total for this street is hero_st + pot_so_far//2
            half_pot = pot_so_far // 2
            new_total = hero_st + half_pot
            if new_total > my_stack:
                # if that exceeds stack, just call it an allin
                new_total = my_stack
            return f"b{new_total}"


# =============================================================================
# 4) PLAY HAND: loops until hand ends
# =============================================================================

def PlayHand(token):
    """
    Plays one hand vs Slumbot, printing detailed info:
     - Street changes
     - Who bet/raised/called/folded
     - Pot updates
     - Stacks at start/end
    """
    # Start new hand
    resp = NewHand(token)
    if 'token' in resp:
        token = resp['token']

    action_str = resp.get('action','')
    client_pos = resp.get('client_pos', 0)  # 0=big blind, 1=small blind
    hole_cards = resp.get('hole_cards', [])
    board      = resp.get('board', [])
    winnings   = resp.get('winnings', None)

    # We'll track "hero_stack" and "villain_stack" from the start:
    hero_stack    = 20000
    villain_stack = 20000

    # Print blinds info
    print("\n====================================")
    print("NEW HAND STARTED")
    print(f"Slumbot Token: {token}")
    print(f"Hero seat={client_pos} -> {'Big Blind' if client_pos==0 else 'Small Blind'}")
    print(f"Initial Stacks: Hero={hero_stack}, Villain={villain_stack}")
    print(f"Hole cards: {hole_cards}")
    print(f"Board: {board if board else 'No board cards yet'}")
    print(f"Action so far: '{action_str}'")

    last_street_idx = 0
    last_hero_street = 100 if client_pos==0 else 50
    last_vill_street = 50 if client_pos==0 else 100
    pot_total_previous = 0

    while True:
        if winnings is not None:
            print(f"Hand ended immediately, winnings={winnings}")
            print("====================================\n")
            return token, winnings

        parsed = parse_action_enhanced(action_str, client_pos)
        if parsed['error']:
            print(f"Parse error: {parsed['error']}")
            print("====================================\n")
            return token, 0
        
        # Street name
        st_name = get_street_name(parsed['street'])
        if parsed['street'] != last_street_idx:
            # We moved to a new street
            print(f"\n--- Moving to {st_name} ---")
            last_street_idx = parsed['street']
        
        # Recompute the difference in each seat's street contribution => that might be new bets
        hero_delta = parsed['hero_street'] - last_hero_street
        vill_delta = parsed['vill_street'] - last_vill_street

        # If hero_delta>0 => hero bet or called
        # If vill_delta>0 => villain bet or called
        # If hero_delta < 0 => error, etc. We'll keep it simple:
        if hero_delta > 0:
            # hero added money
            if hero_delta == 0:
                pass
            else:
                print(f"Hero put in {hero_delta} more chips on {st_name}.")

        if vill_delta > 0:
            print(f"Villain put in {vill_delta} more chips on {st_name}.")

        # Next to act
        seat_act = parsed['next_to_act']
        if seat_act == -1 and parsed['hand_over']:
            # Hand ended
            # pot_total includes the final street's sum
            print("\n--- Hand Over Detected ---")
            final_pot = parsed['pot_total']
            print(f"Final pot: {final_pot}")
            # compute final stacks (if we wanted)
            print("No further actions possible.")
        
        # update stacks:
        # The total in hero_street is how many chips hero has put in on this street => difference from last time is how many we lost from stack
        hero_stack    -= max(0, hero_delta)
        villain_stack -= max(0, vill_delta)

        # store these for next iteration
        last_hero_street = parsed['hero_street']
        last_vill_street = parsed['vill_street']
        pot_total_previous = parsed['pot_total']

        if resp.get('winnings') is not None or parsed['hand_over']:
            # hand ended
            final_win = resp.get('winnings', 0)
            if final_win is not None:
                print(f"** Hand finished, hero's winnings = {final_win}")
                hero_stack += final_win  # if positive => we gained that
                print(f"** Updated hero stack: {hero_stack}, villain stack: {villain_stack}")
            else:
                print("** Hand finished, no 'winnings' in response??")
            print("====================================\n")
            return token, final_win if final_win is not None else 0

        # Otherwise, we choose an action
        print(f"\nStreet: {st_name}, next_to_act = {parsed['next_to_act']}")
        ai_incr = ChooseActionAI(parsed, hole_cards, board, client_pos)
        if not ai_incr:
            # means it's not our turn or we do nothing
            print("No action from hero, possibly Slumbot's turn.")
            print("====================================\n")
            return token, 0

        print(f"Hero's incremental action: '{ai_incr}'")
        resp = Act(token, ai_incr)
        action_str = resp.get('action','')
        board = resp.get('board', board)
        new_token = resp.get('token', token)
        if new_token:
            token = new_token
        winnings = resp.get('winnings', None)

        # Print updated info
        print(f"Updated action string => '{action_str}'")
        if resp.get('board'):
            print(f"Board => {resp['board']}")



# =============================================================================
# 5) MAIN: log in, play multiple hands
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None)
    parser.add_argument('--password', type=str, default=None)
    parser.add_argument('--num_hands', type=int, default=2)
    args = parser.parse_args()
    
    token = None
    if args.username and args.password:
        token = Login(args.username, args.password)
    
    total_winnings = 0
    for h in range(args.num_hands):
        print(f"\nPlaying hand #{h+1} ...")
        token, w = PlayHand(token)
        total_winnings += (w or 0)
    
    print(f"\nDone! After {args.num_hands} hands, total winnings: {total_winnings}")



if __name__ == '__main__':
    main()
