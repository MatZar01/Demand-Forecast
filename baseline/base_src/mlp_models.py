import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )

    def forward(self, X):
        logits = self.model(X)
        return logits
