import torch
from torch import nn


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
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, emb_2, emb_3, X):
        embeddings_2 = self.embedder_2(emb_2)
        embeddings_3 = self.embedder_3(emb_3)

        seq_inp = torch.concatenate([embeddings_2, embeddings_3, X], dim=2).reshape(X.shape[0], -1)

        logits = self.model(seq_inp)
        return logits


class MLP_emb_tl(nn.Module):
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
            nn.Dropout(0.5)
        )

        self.clf = nn.Linear(64, 1)

    def forward(self, emb_2, emb_3, X):
        embeddings_2 = self.embedder_2(emb_2)
        embeddings_3 = self.embedder_3(emb_3)

        seq_inp = torch.concatenate([embeddings_2, embeddings_3, X], dim=2).reshape(X.shape[0], -1)

        logits = self.model(seq_inp)
        logits = self.clf(logits)

        return logits


class MLP_emb_tl_2(nn.Module):
    def __init__(self, input_dim, cat_2_size, cat_3_size, embedding_size):
        super().__init__()

        self.cat_2_size = cat_2_size
        self.cat_3_size = cat_3_size
        self.embed_dim = embedding_size

        self.embedder_2 = nn.Embedding(self.cat_2_size, self.embed_dim)
        self.embedder_3 = nn.Embedding(self.cat_3_size, self.embed_dim)

        input_dim = 165

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
            nn.Dropout(0.5)
        )

        self.clf = nn.Linear(64, 1)

    def forward(self, emb_2, emb_3, X):
        embeddings_2 = self.embedder_2(emb_2)
        embeddings_3 = self.embedder_3(emb_3)

        seq_inp = torch.concatenate([embeddings_2.flatten(1, 2), embeddings_3.flatten(1, 2), X], axis=1)

        logits = self.model(seq_inp)
        logits = self.clf(logits)

        return logits



class MLP_emb_pool(nn.Module):
    def __init__(self, input_dim: int, cat_2_size, cat_3_size, embedding_size):
        super().__init__()

        self.cat_2_size = cat_2_size
        self.cat_3_size = cat_3_size
        self.embed_dim = embedding_size

        self.embedder_2 = nn.Embedding(self.cat_2_size, self.embed_dim)
        self.embedder_3 = nn.Embedding(self.cat_3_size, self.embed_dim)

        input_dim = input_dim * 15

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.clf = nn.Linear(32, 1)

    def forward(self, emb_2, emb_3, X):
        embeddings_2 = self.embedder_2(emb_2)
        embeddings_3 = self.embedder_3(emb_3)

        seq_inp = torch.concatenate([embeddings_2, embeddings_3, X], dim=2).reshape(X.shape[0], -1)

        logits = self.model(seq_inp)
        logits = self.clf(logits)

        return logits


class tl_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.clf = nn.Linear(64, 1)

    def forward(self, fts):

        logits = self.clf(fts)
        return logits


class Conv_1D(nn.Module):
    def __init__(self, input_dim: int, cat_2_size, cat_3_size, embedding_size):
        super().__init__()
        self.cat_2_size = cat_2_size
        self.cat_3_size = cat_3_size
        self.embed_dim = embedding_size

        self.embedder_2 = nn.Embedding(self.cat_2_size, self.embed_dim)
        self.embedder_3 = nn.Embedding(self.cat_3_size, self.embed_dim)

        input_dim = input_dim * 15

        self.model = nn.Sequential(
            nn.Conv1d(1, 8, 15),
            nn.Conv1d(8, 16, 15),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 16, 15),
            nn.Conv1d(16, 32, 15),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 32, 15),
            nn.Conv1d(32, 32, 15),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.AvgPool1d(2),
        )
        self.clf = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, emb_2, emb_3, X):
        embeddings_2 = self.embedder_2(emb_2)
        embeddings_3 = self.embedder_3(emb_3)

        seq_inp = torch.concatenate([embeddings_2, embeddings_3, X], dim=2).reshape(X.shape[0], -1)
        seq_inp = seq_inp.unsqueeze(1)

        logits = self.model(seq_inp)
        logits = logits.flatten(1)
        logits = self.clf(logits)

        return logits
