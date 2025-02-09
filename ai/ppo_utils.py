# AI/ppo_utils.py

import numpy as np
import torch
import torch.nn.functional as F
from siamese_net import logits_to_probs

# Constants
DELTA1 = 3


##################################################
# 1) OLD vs. NEW POLICY RATIO
##################################################

def ratio(old_probs: np.ndarray, new_probs: np.ndarray, action_idx: int):
    """
    PPO ratio = new_probs[action_idx] / old_probs[action_idx].
    We clamp the denominator to avoid divide-by-zero.
    """
    denom = old_probs[action_idx] if old_probs[action_idx] > 1e-10 else 1e-10
    return new_probs[action_idx] / denom


##################################################
# 2) DISCOUNTED RETURNS (r_gamma)
##################################################

def r_gamma(rewards: np.ndarray, gamma: float) -> float:
    """
    Accumulates rewards in reverse order (except the first),
    multiplying by gamma at each step, then adds the first reward.
    """
    r_val = 0.0
    # Reverse except the first element
    for reward in rewards[:0:-1]:
        r_val = gamma * (r_val + reward)
    if len(rewards) > 0:
        r_val += rewards[0]
    return r_val


##################################################
# 3) ADVANTAGE ESTIMATION (a_gae)
##################################################

def a_gae_old(results, states, value_function_fn, gamma=0.999, lambda_=0.99):
    """
    A single-scalar GAE, as in your original code:
      - len(states) == len(results)
      - We do NOT assume an extra 'terminal state' 
    """
    N = len(results)
    if N == 0:
        return 0.0
    
    v0 = value_function_fn(states[0])
    # partial sums S[k] = Σ_{i=0..k-1} gamma^i * results[i]
    S = np.zeros(N+1, dtype=float)
    for i in range(N):
        S[i+1] = S[i] + (gamma ** i) * results[i]
    
    gae_sum = 0.0
    for k in range(1, N):
        a_k = -v0 + S[k] + (gamma ** k) * value_function_fn(states[k])
        gae_sum += (lambda_ ** (k - 1)) * a_k
    return (1 - lambda_) * gae_sum

def a_gae(states, rewards, value_function_fn, gamma=0.999, lambda_=0.99):
    """
    A single-scalar GAE, as in your original code:
      - len(states) == len(results)
      - We do NOT assume an extra 'terminal state' 
    """

    N = len(states)
    if N == 0:
        return 0.0
    
    v0 = value_function_fn(states[0]).item()
    # partial sums S[k] = Σ_{i=0..k-1} gamma^i * results[i]
    S = np.zeros(N+1, dtype=float)
    for i in range(N):
        S[i+1] = S[i] + (gamma ** i) * rewards[i]
    
    gae_sum = 0.0
    for k in range(1, N):
        a_k = -v0 + S[k] + (gamma ** k) * value_function_fn(states[k]).item()
        gae_sum += (lambda_ ** (k - 1)) * a_k
    return (1 - lambda_) * gae_sum


##################################################
# 4) TRINAL-CLIP POLICY LOSS
##################################################

def tc_loss_function(ratio_val: float, advantage: float, epsilon: float):
    """
    ratio is clipped first to [1-epsilon, 1+epsilon],
    then clipped again to deltas[0].
    => min(ratio, clip(ratio,1±eps), delta1) * advantage
    """
    # (delta1, _, _) = deltas
    inner = np.clip(ratio_val, 1 - epsilon, 1 + epsilon)
    outer = np.clip(ratio_val, inner, DELTA1)
    return outer * advantage


##################################################
# 5) CLIPPED VALUE LOSS
##################################################

def v_loss(r_gamma_val: float, state, deltas: tuple, value_state):
    """
    (clip(r_gamma_val,-delta2,delta3) - V(state))^2
    If no value_function_fn is provided, defaults to 0 (legacy).
    """
    # For backward compatibility, fallback to 0 if not provided
    
    clipped_ret = np.clip(r_gamma_val, -deltas[0], deltas[1])
    return (clipped_ret - value_state)**2


##################################################
# 6) GET DELTAS / GET ACTION (utilities)
##################################################


def get_action_from_probs(probs: np.ndarray) -> int:
    """
    Sample an action from a given 1D distribution.
    """
    return np.random.choice(len(probs), p=probs)


##############################################################
# 7) MODEL-BASED VALUE FUNCTION CREATOR
##############################################################

def make_model_value_function_old(model, build_card_rep_fn, build_action_rep_fn):
    """
    Returns a function 'value_fn(state)' that uses the siamese model
    to compute a scalar value for the given 'state'.

    build_card_rep_fn(state): -> CardRepresentation
    build_action_rep_fn(state): -> ActionRepresentation

    In real code, you might store the full environment state for each timestep
    so you don't have to guess how to rebuild reps from 'state'.
    """
    def value_function_impl(state):
        card_rep = build_card_rep_fn(state)
        action_rep = build_action_rep_fn(state)

        # Convert to torch
        card_np = card_rep.card_tensor[np.newaxis,...]
        action_np = action_rep.action_tensor[np.newaxis,...]
        card_t = torch.from_numpy(card_np).float()
        action_t = torch.from_numpy(action_np).float()

        with torch.no_grad():
            _, val_out = model(action_t, card_t)
        return val_out.item()
    
    return value_function_impl


def make_model_value_function(model):
    """
    Returns a function 'value_fn(state)' that uses the siamese model
    to compute a scalar value for the given 'state'.

    build_card_rep_fn(state): -> CardRepresentation
    build_action_rep_fn(state): -> ActionRepresentation
    """
    
    def value_function_impl(state):
        card_np = state[0][np.newaxis,...]
        action_np = state[1][np.newaxis,...]
        card_t = torch.from_numpy(card_np).float()
        action_t = torch.from_numpy(action_np).float()
        _, val_out = model.forward(action_t, card_t) # I dont think we need to use no_grad here. We want to update the model at some point
        return val_out # I return the tensor for doing backpropagation later. I think is the only way.
    
    return value_function_impl

