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


class Embedder_double(nn.Module):
    def __init__(self, cat_2_size, cat_3_size, embded_dim):
        super().__init__()

        self.cat_2_size = cat_2_size
        self.cat_3_size = cat_3_size
        self.embed_dim = embded_dim

        self.embedder_2 = nn.Embedding(self.cat_2_size, self.embed_dim)
        self.embedder_3 = nn.Embedding(self.cat_3_size, self.embed_dim)

        self.model = nn.Sequential(
            nn.Linear(self.embed_dim*2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 3)
        )

    def forward(self, X_2, X_3):
        embeddings_2 = self.embedder_2(X_2).view((-1, self.embed_dim))
        embeddings_3 = self.embedder_3(X_3).view((-1, self.embed_dim))
        embeddings = torch.concatenate([embeddings_2, embeddings_3], dim=1)

        logits = self.model(embeddings)
        return logits
