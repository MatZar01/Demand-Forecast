import torch
from torch import nn


class Embedder(nn.Module):
    def __init__(self, cat_size, embded_dim):
        super().__init__()

        self.cat_size = cat_size
        self.embed_dim = embded_dim

        self.embedder = nn.Embedding(self.cat_size, self.embed_dim)

        self.model = nn.Sequential(
            nn.Linear(self.embed_dim, 32),
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
        embeddings = self.embedder(X).view((-1, self.embed_dim))
        logits = self.model(embeddings)
        return logits
