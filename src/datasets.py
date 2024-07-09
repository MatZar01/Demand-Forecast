import numpy as np
import pandas as pd


class DataSet:
    def __init__(self, paths, recombine_data: list = [0.7, 0.3], shuffle: bool = True, seed: int = 24):
        self.path_train = paths['TRAIN']
        self.path_test = paths['TEST']
        self.name = self.path_train.split('/')[-2]
        self.recombine_data = recombine_data
        self.shuffle = shuffle
        self.seed = seed

        self.train = None
        self.val = None
        self.test = None
        self.data_merged = None
        self.data_unshuffled = None
        self.header = None
        self.header_test = None

        self.load_data()

    def load_data(self):
        # try training data
        try:
            df_train = pd.read_csv(self.path_train)
            self.header = [*df_train]
        except FileNotFoundError:
            df_train = None

        # try testing data
        try:
            df_test = pd.read_csv(self.path_test)
            self.test = df_test.to_numpy()
            self.header_test = [*df_test]
        except FileNotFoundError:
            df_test = None

        self.data_unshuffled = df_train.to_numpy()
        if self.recombine_data:
            self.recombine(df_train)

    def recombine(self, train):
        data = train.to_numpy()
        if self.shuffle:
            indexes = self.get_shuffled_indexes(data)
            data = data[indexes]

        train_len = int(data.shape[0]*self.recombine_data[0])
        self.train = data[:train_len, :]
        self.val = data[train_len:, :]
        self.data_merged = data
        print(f'[INFO] Data:\ntrain: {self.train.shape[0]} points\nval: {self.val.shape[0]}\ntest: {self.test.shape[0]}')
        print(f'Header: {self.header}')

    def get_shuffled_indexes(self, data):
        data_idx = list(range(data.shape[0]))
        np.random.seed(self.seed)
        np.random.shuffle(data_idx)
        return data_idx
