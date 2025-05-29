# AI/siamese_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Autoencoder for the action branch
class ActionAutoencoder(nn.Module):
    def __init__(self, action_in_shape=(24, 4, 9)):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(action_in_shape[0], 32, kernel_size=3, padding=1),  # (B, 32, 4, 9)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                                  # (B, 32, 2, 4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),                    # (B, 64, 2, 4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                                   # (B, 64, 1, 2)
        )
        # Decoder: Reverse the encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),            # (B, 32, 2, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(32, action_in_shape[0], kernel_size=2, stride=2, output_padding=(0, 1)),
            nn.Sigmoid()  # assuming inputs are normalized between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def train_encoder(self, dataloader, num_epochs=10):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=5e-3)
        for epoch in range(num_epochs):
            for action_input in dataloader:
                # Assume batch['action_input'] has shape (B, 24, 4, 9)
                # action_input = batch['action_input']
                action_input = action_input[0]
                print(action_input.shape)
                optimizer.zero_grad()
                reconstructed = self.forward(action_input)
                loss = criterion(reconstructed, action_input)
                loss.backward()
                optimizer.step()
            print(f"Action Encoder Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Autoencoder for the card branch
class CardAutoencoder(nn.Module):
    def __init__(self, card_in_shape=(6, 4, 13)):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(card_in_shape[0], 32, kernel_size=3, padding=1),    # (B, 32, 4, 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                                   # (B, 32, 2, 6)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),                   # (B, 64, 2, 6)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)                                    # (B, 64, 1, 3)
        )
        # Decoder: Reverse the encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),            # (B, 32, 2, 6)
            nn.ReLU(),
            nn.ConvTranspose2d(32, card_in_shape[0], kernel_size=2, stride=2, output_padding=(0, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def train_encoder(self, dataloader, num_epochs=10):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=5e-3)
        for epoch in range(num_epochs):
            for card_input in dataloader:
                # Assume batch['card_input'] has shape (B, 6, 4, 13)
                # card_input = batch['card_input']
                card_input = card_input[0]
                print(card_input.shape)
                optimizer.zero_grad()
                reconstructed = self.forward(card_input)
                loss = criterion(reconstructed, card_input)
                loss.backward()
                optimizer.step()
            print(f"Card Encoder Epoch {epoch+1}, Loss: {loss.item():.4f}")

class PseudoSiameseNet(nn.Module):
    def __init__(self, pretrained_action_encoder, pretrained_card_encoder, hidden_dim=256, num_actions=9):
        super().__init__()
        self.action_encoder = pretrained_action_encoder
        self.card_encoder = pretrained_card_encoder
        
        # Determine flattened sizes (using dummy data)
        with torch.no_grad():
            dummy_action = torch.zeros(1, 24, 4, 9)
            dummy_card = torch.zeros(1, 6, 4, 13)
            a_out = self.action_encoder(dummy_action)
            c_out = self.card_encoder(dummy_card)
            a_flat = a_out.view(1, -1).size(1)
            c_flat = c_out.view(1, -1).size(1)
        
        fusion_in_dim = a_flat + c_flat
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, action_input, card_input):
        a_feat = self.action_encoder(action_input)
        c_feat = self.card_encoder(card_input)
        combined = torch.cat([a_feat.view(a_feat.size(0), -1),
                              c_feat.view(c_feat.size(0), -1)], dim=1)
        fusion = self.fusion_fc(combined)
        policy_logits = self.policy_head(fusion)
        value = self.value_head(fusion)
        return policy_logits, value



class PseudoSiameseNet_OLD(nn.Module):
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
