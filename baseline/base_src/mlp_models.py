import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        logits = self.model(X)
        return logits


class MLP_emb(nn.Module):
    def __init__(self, input_dim: int, cat_2_size, cat_3_size, embedding_size):
        super().__init__()

        self.cat_2_size = cat_2_size
        self.cat_3_size = cat_3_size
        self.embed_dim = embedding_size

        self.embedder_2 = nn.Embedding(self.cat_2_size, self.embed_dim)
        self.embedder_3 = nn.Embedding(self.cat_3_size, self.embed_dim)

        input_dim = input_dim * 15

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, emb_2, emb_3, X):
        embeddings_2 = self.embedder_2(emb_2)
        embeddings_3 = self.embedder_3(emb_3)

        seq_inp = torch.concatenate([embeddings_2, embeddings_3, X], dim=2).reshape(X.shape[0], -1)

        logits = self.model(seq_inp)
        return logits