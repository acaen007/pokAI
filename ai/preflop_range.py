import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
from card_representation import CardRepresentation
from action_representation import ActionRepresentation
from ppo_utils import to_torch_input

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
# ------------------------------------------------------------
# Card encoding
# ------------------------------------------------------------
RANKS = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
    '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
}

SUITS = {
    's': 0,  # Spades
    'c': 1,  # Clubs
    'd': 2,  # Diamonds
    'h': 3   # Hearts
}

# Reverse mapping for display purposes
RANKS_INV = {v: k for k, v in RANKS.items()}


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def softmax(x):
    """Numerically stable softmax."""
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

def dummy_model(action_tensor, card_representation):
    """
    A dummy model function for testing. It computes a heuristic play probability
    based on the normalized ranks of the two hole cards plus a bonus if suited.
    
    Replace this with your model's inference function later.
    """
    (r1, s1), (r2, s2) = card_representation.hole_cards
    norm1 = r1 / 12.0  # Scale: 0 (for 2) to 1 (for Ace)
    norm2 = r2 / 12.0
    base = (norm1 + norm2) / 2.0
    suited_bonus = 0.1 if s1 == s2 else 0.0
    prob_play = min(base + suited_bonus, 0.99)
    play_logit = np.log(prob_play / (1 - prob_play))
    logits = np.array([0.0, play_logit])
    return logits, None

def get_hand_type(card1, card2):
    """
    Returns a string representing the hand type.
      - For pairs: e.g., "AA"
      - For non-pairs: higher card first with a suffix 's' if suited or 'o' if offsuit,
        e.g. "AKs" or "AKo".
    """
    r1, s1 = card1
    r2, s2 = card2
    if r1 == r2:
        return RANKS_INV[r1] + RANKS_INV[r2]
    else:
        # Order so that the higher rank comes first.
        if r1 > r2:
            high, low = (r1, s1), (r2, s2)
        else:
            high, low = (r2, s2), (r1, s1)
        suited_flag = 's' if high[1] == low[1] else 'o'
        return RANKS_INV[high[0]] + RANKS_INV[low[0]] + suited_flag

def get_deck():
    """Return a list of all cards in the deck as (rank, suit) tuples."""
    return [(r, s) for r in range(13) for s in range(4)]

# ------------------------------------------------------------
# Simulation Functions
# ------------------------------------------------------------
def simulate_preflop_range(action_r, model_fn=dummy_model, device='cuda'):
    """
    Simulate the preflop range for every two-card combination in a batched manner.
    
    Parameters:
        action_r: The input tensor or parameter for your model.
        model_fn: A model function (PyTorch module) with a .forward() method that takes (action_batch, card_batch)
                  and returns (policy_logits, value).
        device: Torch device to use.
        
    Returns:
        Three dictionaries with the average play probabilities for:
            - Pocket pairs (avg_pairs)
            - Suited hands (avg_suited)
            - Offsuit hands (avg_offsuit)
            
    Note:
        - All card representations are processed with one forward call in a batch.
        - Only allowed actions are considered: indices [0, 2, 6, 7, 8] where index 0 corresponds to folding.
          The play probability is computed as 1 - (fold probability normalized over these allowed actions).
    """
    import torch
    import itertools
    import numpy as np

    # Put the model in evaluation mode.
    model_fn.eval()
    
    # Build deck and prepare lists for card representations and hand labels.
    deck = get_deck()
    card_reps = []
    hand_types = []
    for card1, card2 in itertools.combinations(deck, 2):
        cr = CardRepresentation()
        cr.set_preflop([card1, card2])
        card_reps.append(cr)
        hand_types.append(get_hand_type(card1, card2))
    
    # Convert each CardRepresentation into torch input via to_torch_input.
    # Note: to_torch_input returns tensors with an added batch dimension.
    action_tensors = []
    card_tensors = []
    print(card_reps)
    for idx, cr in enumerate(card_reps):
        act_tensor, card_tensor = to_torch_input(cr, action_r, device)
        print(f"Hand {get_hand_type(*cr.hole_cards)}: card tensor sum = {card_tensor.sum().item()}")
        # Option 1: If you want to keep the batch dimension from to_torch_input, do not squeeze.
        # Option 2: If you want to remove an extraneous singleton batch dimension, then squeeze.
        # Here I suggest printing out the shapes to decide:
        # print(f"Hand {hand_types[idx]}: act_tensor shape {act_tensor.shape}, card_tensor shape {card_tensor.shape}")
        action_tensors.append(act_tensor.squeeze(0))
        card_tensors.append(card_tensor.squeeze(0))
    
    # Debug: check that card_tensors are actually different.
    for i in range(3):
        print(f"Debug: Hand {hand_types[i]} card tensor sum: {card_tensors[i].sum().item()}")

    # Create batch tensors.
    action_batch = torch.stack(action_tensors, dim=0).to(device)  # shape: (N, ...)
    card_batch = torch.stack(card_tensors, dim=0).to(device)        # shape: (N, ...)
    
    # Forward pass: logits shape is assumed to be (batch_size, num_actions)
    logits, _ = model_fn.forward(action_batch, card_batch)
    
    # Restrict logits to the allowed actions: [0, 2, 6, 7, 8] (with index 0 as fold).
    allowed = [0, 2, 6, 7, 8]
    logits_allowed = logits[:, allowed]  # shape: (N, 5)
    
    # Compute softmax over the allowed actions.
    probs_allowed = torch.nn.functional.softmax(logits_allowed, dim=-1)
    probs_allowed = probs_allowed.cpu().detach().numpy()  # shape: (N, 5)
    
    # The fold probability is at index 0; thus play probability is 1 - fold probability.
    fold_probs = probs_allowed[:, 0]
    play_probs = 1.0 - fold_probs
    
    # Aggregate play probabilities by hand type.
    pairs = {}
    suited = {}
    offsuit = {}
    for i, hand in enumerate(hand_types):
        play_prob = play_probs[i]
        if len(hand) == 2:  # Pocket pair (e.g., "AA")
            pairs.setdefault(hand, []).append(play_prob)
        else:
            if hand[-1] == 's':
                suited.setdefault(hand, []).append(play_prob)
            else:
                offsuit.setdefault(hand, []).append(play_prob)
    
    avg_pairs   = {k: np.mean(v) for k, v in pairs.items()}
    avg_suited  = {k: np.mean(v) for k, v in suited.items()}
    avg_offsuit = {k: np.mean(v) for k, v in offsuit.items()}
    
    return avg_pairs, avg_suited, avg_offsuit


def build_merged_matrix(avg_pairs, avg_suited, avg_offsuit):
    """
    Build a single merged 13x13 matrix with the play probabilities.
    
    Conventions:
        - Diagonal: Pocket pairs.
        - Upper triangle: Suited hands.
        - Lower triangle: Offsuit hands.
    
    Returns:
        merged_matrix: A 13x13 numpy array.
        annotations: A 13x13 list with hand labels.
        rank_labels: Rank labels (descending order) for axis ticks.
    """
    # Descending order: Ace (12) to 2 (0)
    ranks_order = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    rank_labels = [RANKS_INV[r] for r in ranks_order]
    merged_matrix = np.full((13, 13), np.nan)
    annotations = [['' for _ in range(13)] for _ in range(13)]
    
    for i, r1 in enumerate(ranks_order):
        for j, r2 in enumerate(ranks_order):
            if r1 == r2:
                hand = RANKS_INV[r1] + RANKS_INV[r2]
                value = avg_pairs.get(hand, np.nan)
                merged_matrix[i, j] = value
                annotations[i][j] = hand
            elif i < j:
                # Upper triangle: suited hands.
                hand = RANKS_INV[r1] + RANKS_INV[r2] + 's'
                value = avg_suited.get(hand, np.nan)
                merged_matrix[i, j] = value
                annotations[i][j] = hand
            else:
                # Lower triangle: offsuit hands.
                hand = RANKS_INV[r2] + RANKS_INV[r1] + 'o'
                value = avg_offsuit.get(hand, np.nan)
                merged_matrix[i, j] = value
                annotations[i][j] = hand
                
    return merged_matrix, annotations, rank_labels

def plot_merged_range(merged_matrix, annotations, rank_labels, title="Merged Preflop Opening Range (Play Probability)"):
    """
    Plot the merged range chart with annotations.
    
    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(merged_matrix, cmap='viridis', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(rank_labels)))
    ax.set_yticks(np.arange(len(rank_labels)))
    ax.set_xticklabels(rank_labels)
    ax.set_yticklabels(rank_labels)
    ax.set_xlabel("Second Card")
    ax.set_ylabel("First Card")
    ax.set_title(title)
    
    # Annotate each cell with the hand label and play probability.
    for i in range(len(rank_labels)):
        for j in range(len(rank_labels)):
            if not np.isnan(merged_matrix[i, j]):
                text = f"{annotations[i][j]}\n{merged_matrix[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="w", fontsize=8)
                
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig, ax

def plot_preflop_range(model_fn=dummy_model, show=True):
    """
    High-level function to simulate, build, and plot the merged preflop range.
    
    Parameters:
        action_tensor: Input tensor for the model.
        model_fn: Function that computes (policy_logits, value) from (CardRepresentation, action_tensor).
        show: Whether to call plt.show() to display the plot.
        
    Returns:
        fig: The matplotlib figure.
        merged_matrix: The computed 13x13 matrix.
        annotations: The annotations for each cell.
    """
    action_tensor = ActionRepresentation(rounds=4, max_actions_per_round=6, nb=9).action_tensor # CHECK THIS. WE SHOULD INCLUDE THE LEGAL MOVES
    avg_pairs, avg_suited, avg_offsuit = simulate_preflop_range(action_tensor, model_fn)
    merged_matrix, annotations, rank_labels = build_merged_matrix(avg_pairs, avg_suited, avg_offsuit)
    fig, ax = plot_merged_range(merged_matrix, annotations, rank_labels)
    
    if show:
        plt.show()
    
    return fig, merged_matrix, annotations

# Example usage (in your notebook):
if __name__ == "__main__":
    # Replace 'action_tensor' with your actual tensor when available.
    plot_preflop_range()
