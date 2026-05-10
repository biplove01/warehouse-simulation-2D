# model.py
import torch.nn as nn


class QNetwork(nn.Module):
    """Dueling DQN Architecture — shared by Agent 1 and Agent 2."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared_features = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
        )
        self.advantage_stream = nn.Linear(128, action_dim)
        self.value_stream      = nn.Linear(128, 1)

    def forward(self, state):
        features           = self.shared_features(state)
        state_value        = self.value_stream(features)
        action_advantages  = self.advantage_stream(features)
        return state_value + (
            action_advantages - action_advantages.mean(dim=1, keepdim=True)
        )