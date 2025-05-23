# AI/siamese_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PseudoSiameseNet(nn.Module):
    """
    Convolutional branches for:
      - Action tensor (shape=(24,4,9))
      - Card tensor   (shape=(6,4,13))
    Merged, then output policy_logits & value.
    """
    def __init__(
        self,
        action_in_shape=(24, 4, 9),
        card_in_shape=(6, 4, 13),
        conv_out_dim=128,
        hidden_dim=256,
        num_actions=9
    ):
        super().__init__()
        
        # Action branch
        self.action_conv = nn.Sequential(
            nn.Conv2d(action_in_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Card branch
        self.card_conv = nn.Sequential(
            nn.Conv2d(card_in_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Determine flatten sizes
        with torch.no_grad():
            dummy_action = torch.zeros(1, action_in_shape[0], action_in_shape[1], action_in_shape[2])
            dummy_card   = torch.zeros(1, card_in_shape[0],   card_in_shape[1],   card_in_shape[2])
            a_out = self.action_conv(dummy_action)
            c_out = self.card_conv(dummy_card)
            self.a_flat = a_out.view(1, -1).size(1)
            self.c_flat = c_out.view(1, -1).size(1)
        
        self.action_fc = nn.Sequential(
            nn.Linear(self.a_flat, conv_out_dim),
            nn.ReLU()
        )
        self.card_fc = nn.Sequential(
            nn.Linear(self.c_flat, conv_out_dim),
            nn.ReLU()
        )
        
        fusion_in_dim = conv_out_dim * 2
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Final heads
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head  = nn.Linear(hidden_dim, 1)
    
    def forward(self, action_input, card_input):
        """
        action_input: shape (B,24,4,9)
        card_input:   shape (B,6,4,13)
        """
        x_a = self.action_conv(action_input)
        x_a = x_a.view(x_a.size(0), -1)
        x_a = self.action_fc(x_a)
        
        x_c = self.card_conv(card_input)
        x_c = x_c.view(x_c.size(0), -1)
        x_c = self.card_fc(x_c)
        
        x = torch.cat([x_a, x_c], dim=1)
        x = self.fusion_fc(x)
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

def logits_to_probs(logits):
    return F.softmax(logits, dim=-1)

def clone_model_weights(src_model: nn.Module, dst_model: nn.Module):
    """
    Copies parameters from src_model to dst_model (in-place).
    This function is used to create an 'old policy' from a 'new policy'.
    """
    dst_model.load_state_dict(src_model.state_dict())

# def to_torch_input(card_input, action_input):
#     """
#     Converts a card representation and an action representation to torch tensors.
    
#     Args:
#         card_input: Either a CardRepresentation object or a numpy array representing the card tensor.
#         action_input: Either an ActionRepresentation object or a numpy array representing the action tensor.
        
#     Returns:
#         A tuple (action_t, card_t) of torch tensors.
#     """
#     import numpy as np
#     import torch
    
#     # If card_input is already a numpy array, use it directly; otherwise, assume it has a card_tensor attribute.
#     if isinstance(card_input, np.ndarray):
#         card_np = card_input[np.newaxis, ...]
#     else:
#         card_np = card_input.card_tensor[np.newaxis, ...]
    
#     if isinstance(action_input, np.ndarray):
#         action_np = action_input[np.newaxis, ...]
#     else:
#         action_np = action_input.action_tensor[np.newaxis, ...]
    
#     card_t = torch.from_numpy(card_np).float()
#     action_t = torch.from_numpy(action_np).float()
#     return action_t, card_t

def to_torch_input(card_input, action_input, device):
    import numpy as np, torch
    if isinstance(card_input, np.ndarray):
        c_np = card_input[None]
    else:
        c_np = card_input.card_tensor[None]
    if isinstance(action_input, np.ndarray):
        a_np = action_input[None]
    else:
        a_np = action_input.action_tensor[None]

    card_t = torch.from_numpy(c_np).float().to(device)
    action_t = torch.from_numpy(a_np).float().to(device)
    return action_t, card_t