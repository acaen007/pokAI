# AI/ppo_utils.py

import numpy as np
import torch
import torch.nn.functional as F
from siamese_net import logits_to_probs
from debug_utils import debug_print

# Constants
DELTA1 = 3

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

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
    Compute multi-step return R = r[0] + gamma*r[1] + gamma^2*r[2] + ...
    """
    R = 0.0
    for reward in rewards[::-1]:
        R = reward + gamma * R
    return R

##################################################
# 3) ADVANTAGE ESTIMATION (GAE)
##################################################

def a_gae(states, rewards, value_fn, gamma=0.999, lambda_=0.99) -> float:
    """
    Generalized Advantage Estimation (single-scalar) per AlphaHold'em.
    """
    N = len(states)
    if N == 0:
        return 0.0
    # value at t=0
    v0 = value_fn(states[0]).item()
    # prefix sums of discounted rewards
    S = np.zeros(N+1, dtype=float)
    for i in range(N):
        S[i+1] = S[i] + (gamma**i) * rewards[i]
    # GAE sum
    gae = 0.0
    for k in range(1, N):
        vk = value_fn(states[k]).item()
        delta = -v0 + S[k] + (gamma**k) * vk
        gae += (lambda_**(k-1)) * delta
    return (1 - lambda_) * gae


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
    
    debug_print("The deltas are:", deltas)
    debug_print("r_gamma_val:", r_gamma_val, "Value state:", value_state)
    clipped_ret = np.clip(r_gamma_val, -deltas[0], deltas[1])
    debug_print("Clipped return:", clipped_ret, "Value state:", value_state)
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
        card_np, action_np = state
        # debug_print("Before:", card_np.shape)
        action_t, card_t = to_torch_input(card_np, action_np, device)
        # debug_print("After:", card_np.shape)
        _, val_out = model.forward(action_t, card_t) # I dont think we need to use no_grad here. We want to update the model at some point
        return val_out # I return the tensor for doing backpropagation later. I think is the only way.
    
    return value_function_impl

def to_torch_input(card_input, action_input, device):
    """
    Converts a card representation and an action representation to torch tensors.
    
    Args:
        card_input: Either a CardRepresentation object or a numpy array representing the card tensor.
        action_input: Either an ActionRepresentation object or a numpy array representing the action tensor.
        
    Returns:
        A tuple (action_t, card_t) of torch tensors.
    """
    import numpy as np
    import torch
    
    # If card_input is already a numpy array, use it directly; otherwise, assume it has a card_tensor attribute.
    if isinstance(card_input, np.ndarray):
        card_np = card_input[np.newaxis, ...]
    else:
        card_np = card_input.card_tensor[np.newaxis, ...]
    
    if isinstance(action_input, np.ndarray):
        action_np = action_input[np.newaxis, ...]
    else:
        action_np = action_input.action_tensor[np.newaxis, ...]
    
    card_t = torch.from_numpy(card_np).float().to(device)
    action_t = torch.from_numpy(action_np).float().to(device)
    return action_t, card_t

