import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
import pickle


class Embedding_dataset(Dataset):
    def __init__(self, path, column, train: bool = True, label_encoders = None, out_path: str = None):
        self.train = train
        self.column = column
        self.path = path
        self.name = path.split('/')[-1].split('.')[0]
        self.data_all = None
        self.label_encoders = label_encoders
        self.out_path = out_path

        self.X = None
        self.labels = None

        self.load_data()

        data_year = self.split_to_years(self.data_all)
        self.X_uncoded = self.get_relevant_column(data_year)

        # onehot X data
        self.X = self.onehot_score(self.X_uncoded)
        self.data_shape = self.X[0].shape[-1], self.X[1].shape[-1]

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
        self.labels = data[:, -1]
        return data[:, self.column]

    def onehot_score(self, X_tensor) -> np.array:
        if self.label_encoders is None:
            self.label_encoders = []
            for i in range(len(self.column)):
                self.label_encoders.append(OneHotEncoder())
                self.label_encoders[i].fit(X_tensor[:, i].reshape(-1, 1))

                if self.out_path is not None:
                    pickle.dump(self.label_encoders[i], open(f'{self.out_path}/onehot_C{self.column[i]}.pkl', 'wb'))

        outs = []
        for i in range(len(self.column)):
            outs.append(torch.Tensor(self.label_encoders[i].transform(X_tensor[:, i].reshape(-1, 1)).toarray().astype(int)))
        return outs

    def get_nearest(self, idx, prev: bool):
        m = self.X_uncoded[idx]
        if prev:
            search_space = self.X_uncoded[:idx, :]
            search_index = -1
        else:
            search_space = self.X_uncoded[idx:, :]
            search_index = 0
        try:
            label = self.labels[np.argwhere((search_space == m).all(axis=1))[search_index]].item()
        except:
            label = self.labels[idx]
        return label

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        prev_value = self.get_nearest(idx, prev=True)
        next_value = self.get_nearest(idx, prev=False)
        return torch.LongTensor([np.argmax(self.X[0][idx])]), torch.LongTensor([np.argmax(self.X[1][idx])]), \
            torch.Tensor([prev_value, self.labels[idx], next_value])

