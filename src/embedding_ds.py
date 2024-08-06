import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch


class Embedding_dataset(Dataset):
    def __init__(self, path, column, train: bool = True, label_encoder = None):
        self.train = train
        self.column = column
        self.path = path
        self.name = path.split('/')[-1].split('.')[0]
        self.data_all = None
        self.label_encoder = label_encoder

        self.X = None
        self.labels = None

        self.load_data()

        data_year = self.split_to_years(self.data_all)
        self.get_relevant_column(data_year)

        # onehot X data
        self.X = self.onehot_score(self.X)
        self.data_shape = self.X.shape[-1]

    def load_data(self):
        self.data_all = pd.read_csv(self.path)

    def split_to_years(self, data):
        data = data.to_numpy()
        years = np.array([int(x.split('/')[-1]) for x in data[:, 1]])
        years = years - np.min(years)

        train_y1 = data[np.where(years == 0)]
        train_y2 = data[np.where(years == 1)]
        val = data[np.where(years == 2)]
        train_all = np.vstack([train_y1, train_y2])
        if self.train:
            return train_all
        else:
            return val

    def get_relevant_column(self, data):
        self.X = data[:, self.column]
        self.labels = data[:, -1]

    def onehot_score(self, X_tensor) -> np.array:
        if self.label_encoder is None:
            self.label_encoder = OneHotEncoder()
            self.label_encoder.fit(X_tensor.reshape(-1, 1))
        return self.label_encoder.transform(X_tensor.reshape(-1, 1)).toarray().astype(int)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return torch.LongTensor([np.argmax(self.X[idx])]), torch.Tensor([self.labels[idx]])
