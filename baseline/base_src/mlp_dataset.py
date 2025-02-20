from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.preprocessing import OrdinalEncoder


class MLP_dataset(Dataset):
    def __init__(self, path: str, train: bool, lag: int, get_quant: bool = False, normalize: bool = True,
                 embedders: dict = None, matches: np.ndarray = None):
        self.path = path
        self.train = train
        self.lag = lag
        self.embedders = embedders
        self.matches = matches
        self.normalize = normalize

        self.columns = [4, 5, 6, 7]

        if self.embedders is not None:
            self.onehot_2 = pickle.load(open(embedders['C2']['onehot'], 'rb'))
            self.onehot_3 = pickle.load(open(embedders['C3']['onehot'], 'rb'))
            self.embedder_2 = torch.load(embedders['C2']['cat2vec'])
            self.embedder_3 = torch.load(embedders['C3']['cat2vec'])

            self.columns = [2, 3, 4, 5, 6, 7]

        self.get_quant = get_quant
        if self.get_quant:
            self.columns.append(8)

        self.data = self.load_data()
        self.data_all = self.split_to_years(self.data)

        if self.matches is not None:
            self.data_all = self.get_match()

        self.X, self.y = self.get_x_y()
        self.X_lag, self.y_lag = self.get_lagged_data(self.X, self.y, self.lag)

        sample_item = self.__getitem__(0)
        self.input_shape = sample_item[0].shape[0]

    def encode_cats(self, batch):
        X, y = batch
        col_2_ins = X[0::len(self.columns)]
        col_3_ins = X[1::len(self.columns)]
        onehot_2 = self.onehot_2.transform(col_2_ins.reshape(-1, 1).astype(np.object_)).toarray()
        onehot_3 = self.onehot_3.transform(col_3_ins.reshape(-1, 1).astype(np.object_)).toarray()

        emb_2 = self.embedder_2(torch.LongTensor(np.argmax(onehot_2, axis=1))).detach().numpy()
        emb_3 = self.embedder_2(torch.LongTensor(np.argmax(onehot_3, axis=1))).detach().numpy()

        X = X.reshape(self.lag, -1)
        X = X[:, 2:]
        out = np.hstack([emb_2, emb_3, X]).flatten()

        return out, y

    def get_match(self):
        store_match_train = self.data_all[np.where(self.data_all[:, 2] == self.matches[0])[0]]
        train_single = store_match_train[np.where(store_match_train[:, 3] == self.matches[1])[0]]
        return train_single

    def get_x_y(self):
        return self.data_all[:, self.columns], self.data_all[:, -1]

    def get_lagged_data(self, X, y, lag):
        X_lagged = []
        y_lagged = []
        for i in range(X.shape[0] - (lag - 1) - 1):
            X_lagged.append(X[i:i + lag, :])
            y_lagged.append(y[i + lag])
        return np.array(X_lagged).reshape(len(X_lagged), -1), np.array(y_lagged)

    def load_data(self):
        return pd.read_csv(self.path)

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

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __len__(self):
        return self.X_lag.shape[0]

    def __getitem__(self, idx):
        batch = self.X_lag[idx].astype(float), self.y_lag[idx].astype(float)
        if self.embedders is not None:
            batch = self.encode_cats(batch)
        if self.normalize:
            batch = (self.normalize_data(batch[0]), batch[1])
        return torch.Tensor(batch[0]), torch.Tensor([batch[1]])


class MLP_dataset_emb(Dataset):
    def __init__(self, path: str, train: bool, lag: int, get_quant: bool = False, normalize: bool = True,
                 embedders: dict = None, matches: np.ndarray = None):
        self.path = path
        self.train = train
        self.lag = lag
        self.embedders = embedders
        self.matches = matches
        self.normalize = normalize
        self.cat_2_size = None
        self.cat_3_size = None

        self.onehot_2 = pickle.load(open(embedders['C2']['onehot'], 'rb'))
        self.onehot_3 = pickle.load(open(embedders['C3']['onehot'], 'rb'))

        self.columns = [2, 3, 4, 5, 6, 7]

        self.get_quant = get_quant
        if self.get_quant:
            self.columns.append(8)

        self.data = self.load_data()
        self.data_all = self.split_to_years(self.data)

        if self.matches is not None:
            self.data_all = self.get_match()

        self.X, self.y = self.get_x_y()
        self.X_lag, self.y_lag = self.get_lagged_data(self.X, self.y, self.lag)

        sample_item = self.__getitem__(0)
        self.input_shape = sample_item[0].shape[0]

    def encode_cats(self, batch):
        X, y = batch
        col_2_ins = X[0::len(self.columns)]
        col_3_ins = X[1::len(self.columns)]
        onehot_2 = self.onehot_2.transform(col_2_ins.reshape(-1, 1).astype(np.object_)).toarray()
        onehot_3 = self.onehot_3.transform(col_3_ins.reshape(-1, 1).astype(np.object_)).toarray()
        onehot_2 = np.argmax(onehot_2, axis=1)
        onehot_3 = np.argmax(onehot_3, axis=1)

        self.cat_2_size = len(self.onehot_2.categories_[0])
        self.cat_3_size = len(self.onehot_3.categories_[0])

        X = X.reshape(self.lag, -1)
        X = X[:, 2:]

        return onehot_2, onehot_3, X, y

    def get_match(self):
        store_match_train = self.data_all[np.where(self.data_all[:, 2] == self.matches[0])[0]]
        train_single = store_match_train[np.where(store_match_train[:, 3] == self.matches[1])[0]]
        return train_single

    def get_x_y(self):
        return self.data_all[:, self.columns], self.data_all[:, -1]

    def get_lagged_data(self, X, y, lag):
        X_lagged = []
        y_lagged = []
        for i in range(X.shape[0] - (lag - 1) - 1):
            X_lagged.append(X[i:i + lag, :])
            y_lagged.append(y[i + lag])
        return np.array(X_lagged).reshape(len(X_lagged), -1), np.array(y_lagged)

    def load_data(self):
        return pd.read_csv(self.path)

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

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __len__(self):
        return self.X_lag.shape[0]

    def __getitem__(self, idx):
        batch = self.X_lag[idx].astype(float), self.y_lag[idx].astype(float)
        emb_2, emb_3, X, y = self.encode_cats(batch)
        if self.normalize:
            X = self.normalize_data(X)
        return torch.LongTensor(emb_2), torch.LongTensor(emb_3), torch.Tensor(X), torch.Tensor([y])


class MLP_dataset_emb_2(Dataset):
    def __init__(self, path: str, train: bool, lag: int, get_quant: bool = False, normalize: bool = True,
                 embedders: dict = None, matches: np.ndarray = None):
        self.path = path
        self.train = train
        self.lag = lag
        self.embedders = embedders
        self.matches = matches
        self.normalize = normalize
        matches_all = get_matches(path)
        self.cat_2_size = np.unique(matches_all[:, 0]).size + 1
        self.cat_3_size = np.unique(matches_all[:, 1]).size + 1

        self.data = self.load_data()
        self.data_all = self.split_to_years(self.data)

        if self.matches is not None:
            self.data_all = self.get_match()

        self.X, self.y = self.get_x_y()
        self.X_lag, self.y_lag = self.get_lagged_data(self.X, self.y, self.lag)

        self.sample_item = self.__getitem__(0)
        self.input_shape = self.sample_item[0].shape[0]

    def encode_cats(self, batch):
        X, y = batch
        onehot_2 = X[0::3]
        onehot_3 = X[1::3]
        X = X[2::3]

        return onehot_2, onehot_3, X, y

    def get_match(self):
        store_match_train = self.data_all[np.where(self.data_all[:, 1] == self.matches[0])[0]]
        train_single = store_match_train[np.where(store_match_train[:, 2] == self.matches[1])[0]]
        return train_single

    def get_x_y(self):
        return self.data_all[:, [1, 2, -1]], self.data_all[:, -1]

    def get_lagged_data(self, X, y, lag):
        X_lagged = []
        y_lagged = []
        for i in range(X.shape[0] - (lag - 1) - 1):
            X_lagged.append(X[i:i + lag, :])
            y_lagged.append(y[i + lag])
        return np.array(X_lagged).reshape(len(X_lagged), -1), np.array(y_lagged)

    def load_data(self):
        return pd.read_csv(self.path)

    def split_to_years(self, data):
        data = data.to_numpy()
        years = np.array([int(x.split('-')[-0]) for x in data[:, 0]])

        train_all = data[np.where(years < 2016)]
        val = data[np.where(years >= 2016)]
        if self.train:
            return train_all
        else:
            return val

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __len__(self):
        return self.X_lag.shape[0]

    def __getitem__(self, idx):
        batch = self.X_lag[idx].astype(float), self.y_lag[idx].astype(float)
        emb_2, emb_3, X, y = self.encode_cats(batch)
        return torch.LongTensor(emb_2), torch.LongTensor(emb_3), torch.Tensor(X), torch.Tensor([y])


class MLP_dataset_emb_3(Dataset):
    def __init__(self, path: str, train: bool, lag: int, get_quant: bool = False, normalize: bool = True,
                 embedders: dict = None, matches: np.ndarray = None):
        self.path = path
        self.train = train
        self.lag = lag
        self.embedders = embedders
        self.matches = matches
        self.normalize = normalize
        matches_all = get_matches(path, encode=True)
        self.cat_2_size = np.unique(matches_all[:, 0]).size + 1
        self.cat_3_size = np.unique(matches_all[:, 1]).size + 1

        self.data = self.load_data()
        # swap to encoded
        self.data = self.encode_whole(self.data)

        self.data_all = self.split_to_years(self.data)

        if self.matches is not None:
            self.data_all = self.get_match()

        self.X, self.y = self.get_x_y()
        self.X_lag, self.y_lag = self.get_lagged_data(self.X, self.y, self.lag)

        self.sample_item = self.__getitem__(0)
        self.input_shape = self.sample_item[0].shape[0]

    def encode_whole(self, data):
        data = data.to_numpy()
        wh_encoder = OrdinalEncoder()
        whs = data[:, 1]
        whs_enc = wh_encoder.fit_transform(whs.reshape(-1, 1))

        prod_encoder = OrdinalEncoder()
        prods = data[:, 2]
        prods_enc = prod_encoder.fit_transform(prods.reshape(-1, 1))

        data[:, 1] = whs_enc.flatten()
        data[:, 2] = prods_enc.flatten()

        #data[:, -1] = data[:, -1] / np.max(data[:, -1])
        return data

    def encode_cats(self, batch):
        X, y = batch
        onehot_2 = X[0::3]
        onehot_3 = X[1::3]
        X = X[2::3]

        return onehot_2, onehot_3, X, y

    def get_match(self):
        store_match_train = self.data_all[np.where(self.data_all[:, 1] == self.matches[0])[0]]
        train_single = store_match_train[np.where(store_match_train[:, 2] == self.matches[1])[0]]
        return train_single

    def get_x_y(self):
        return self.data_all[:, [1, 2, -1]], self.data_all[:, -1]

    def get_lagged_data(self, X, y, lag):
        X_lagged = []
        y_lagged = []
        for i in range(X.shape[0] - (lag - 1) - 1):
            X_lagged.append(X[i:i + lag, :])
            y_lagged.append(y[i + lag])
        return np.array(X_lagged).reshape(len(X_lagged), -1), np.array(y_lagged)

    def load_data(self):
        df_train = pd.read_csv(self.path)
        df_train = df_train.drop(df_train.loc[df_train.isnull().any(axis=1)].index)
        return df_train

    def split_to_years(self, data):
        years = np.array([int(x.split('/')[0]) for x in data[:, 3]])

        train_all = data[np.where(years < 2015)]
        val = data[np.where(years >= 2015)]
        if self.train:
            return train_all
        else:
            return val

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __len__(self):
        return self.X_lag.shape[0]

    def __getitem__(self, idx):
        batch = self.X_lag[idx].astype(float), self.y_lag[idx].astype(float)
        emb_2, emb_3, X, y = self.encode_cats(batch)
        return torch.LongTensor(emb_2), torch.LongTensor(emb_3), torch.Tensor(X), torch.Tensor([y])


class MLP_dataset_emb_cluster_2(Dataset):
    def __init__(self, path: str, train: bool, lag: int, get_quant: bool = False, normalize: bool = True,
                 embedders: dict = None, matches: np.ndarray = None):
        self.path = path
        self.train = train
        self.lag = lag
        self.embedders = embedders
        self.matches = matches
        self.normalize = normalize
        matches_all = get_matches(path)
        self.cat_2_size = np.unique(matches_all[:, 0]).size + 1
        self.cat_3_size = np.unique(matches_all[:, 1]).size + 1

        self.data = self.load_data()
        self.data_all = self.split_to_years(self.data)

        if self.matches is not None:
            self.data_all = self.get_match()

        self.X, self.y = self.get_x_y()
        self.X_lag, self.y_lag = self.get_lagged_data(self.X, self.y, self.lag)

        self.sample_item = self.__getitem__(0)
        self.input_shape = self.sample_item[0].shape[0]

    def encode_cats(self, batch):
        X, y = batch
        onehot_2 = X[0::3]
        onehot_3 = X[1::3]
        X = X[2::3]

        return onehot_2, onehot_3, X, y

    def get_match(self):
        train_matches = []
        for match in self.matches:
            try:
                store_match_train = self.data_all[np.where(self.data_all[:, 1] == match[0])[0]]
                train_single = store_match_train[np.where(store_match_train[:, 2] == match[1])[0]]
                if train_single.size != 0:
                    train_matches.append(train_single)
            except ValueError:
                print(f'[INFO] No match for {match} found :-(')
        return train_matches

    def get_x_y(self):
        X_all = []
        y_all = []
        for d in self.data_all:
            X_all.append(d[:, [1, 2, -1]])
            y_all.append(d[:, -1])
        return X_all, y_all

    def get_lagged_data(self, X, y, lag):
        X_lagged = []
        y_lagged = []
        for n in range(len(X)):
            for i in range(X[n].shape[0] - (lag - 1) - 1):
                X_lagged.append(X[n][i:i + lag, :])
                y_lagged.append(y[n][i + lag])
        return np.array(X_lagged).reshape(len(X_lagged), -1), np.array(y_lagged)

    def load_data(self):
        return pd.read_csv(self.path)

    def split_to_years(self, data):
        data = data.to_numpy()
        years = np.array([int(x.split('-')[-0]) for x in data[:, 0]])

        train_all = data[np.where(years < 2016)]
        val = data[np.where(years >= 2016)]
        if self.train:
            return train_all
        else:
            return val

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __len__(self):
        return self.X_lag.shape[0]

    def __getitem__(self, idx):
        batch = self.X_lag[idx].astype(float), self.y_lag[idx].astype(float)
        emb_2, emb_3, X, y = self.encode_cats(batch)
        return torch.LongTensor(emb_2), torch.LongTensor(emb_3), torch.Tensor(X), torch.Tensor([y])


class MLP_dataset_emb_cluster_3(Dataset):
    def __init__(self, path: str, train: bool, lag: int, get_quant: bool = False, normalize: bool = True,
                 embedders: dict = None, matches: np.ndarray = None):
        self.path = path
        self.train = train
        self.lag = lag
        self.embedders = embedders
        self.matches = matches
        self.normalize = normalize
        matches_all = get_matches(path, encode=True)
        self.cat_2_size = np.unique(matches_all[:, 0]).size + 1
        self.cat_3_size = np.unique(matches_all[:, 1]).size + 1

        self.data = self.load_data()
        # swap to encoded
        self.data = self.encode_whole(self.data)

        self.data_all = self.split_to_years(self.data)

        if self.matches is not None:
            self.data_all = self.get_match()

        self.X, self.y = self.get_x_y()
        self.X_lag, self.y_lag = self.get_lagged_data(self.X, self.y, self.lag)

        self.sample_item = self.__getitem__(0)
        self.input_shape = self.sample_item[0].shape[0]

    def encode_whole(self, data):
        data = data.to_numpy()
        wh_encoder = OrdinalEncoder()
        whs = data[:, 1]
        whs_enc = wh_encoder.fit_transform(whs.reshape(-1, 1))

        prod_encoder = OrdinalEncoder()
        prods = data[:, 2]
        prods_enc = prod_encoder.fit_transform(prods.reshape(-1, 1))

        data[:, 1] = whs_enc.flatten()
        data[:, 2] = prods_enc.flatten()

        #data[:, -1] = data[:, -1] / np.max(data[:, -1])
        return data

    def encode_cats(self, batch):
        X, y = batch
        onehot_2 = X[0::3]
        onehot_3 = X[1::3]
        X = X[2::3]

        return onehot_2, onehot_3, X, y

    def get_match(self):
        train_matches = []
        for match in self.matches:
            try:
                store_match_train = self.data_all[np.where(self.data_all[:, 1] == match[0])[0]]
                train_single = store_match_train[np.where(store_match_train[:, 2] == match[1])[0]]
                if train_single.size != 0:
                    train_matches.append(train_single)
            except ValueError:
                print(f'[INFO] No match for {match} found :-(')
        return train_matches

    def get_x_y(self):
        X_all = []
        y_all =[]
        for d in self.data_all:
            X_all.append(d[:, [1, 2, -1]])
            y_all.append(d[:, -1])
        return X_all, y_all

    def get_lagged_data(self, X, y, lag):
        X_lagged = []
        y_lagged = []
        for n in range(len(X)):
            for i in range(X[n].shape[0] - (lag - 1) - 1):
                X_lagged.append(X[n][i:i + lag, :])
                y_lagged.append(y[n][i + lag])
        return np.array(X_lagged).reshape(len(X_lagged), -1), np.array(y_lagged)

    def load_data(self):
        df_train = pd.read_csv(self.path)
        df_train = df_train.drop(df_train.loc[df_train.isnull().any(axis=1)].index)
        return df_train

    def split_to_years(self, data):
        years = np.array([int(x.split('/')[0]) for x in data[:, 3]])

        train_all = data[np.where(years < 2015)]
        val = data[np.where(years >= 2015)]
        if self.train:
            return train_all
        else:
            return val

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __len__(self):
        return self.X_lag.shape[0]

    def __getitem__(self, idx):
        batch = self.X_lag[idx].astype(float), self.y_lag[idx].astype(float)
        emb_2, emb_3, X, y = self.encode_cats(batch)
        return torch.LongTensor(emb_2), torch.LongTensor(emb_3), torch.Tensor(X), torch.Tensor([y])


class MLP_dataset_cluster(Dataset):
    def __init__(self, path: str, train: bool, lag: int, get_quant: bool = False, normalize: bool = True,
                 embedders: dict = None, matches: np.ndarray = None):
        self.path = path
        self.train = train
        self.lag = lag
        self.embedders = embedders
        self.matches = matches
        self.normalize = normalize
        self.cat_2_size = None
        self.cat_3_size = None

        self.columns = [4, 5, 6, 7]

        if self.embedders is not None:
            self.onehot_2 = pickle.load(open(embedders['C2']['onehot'], 'rb'))
            self.onehot_3 = pickle.load(open(embedders['C3']['onehot'], 'rb'))

            self.columns = [2, 3, 4, 5, 6, 7]

        self.get_quant = get_quant
        if self.get_quant:
            self.columns.append(8)

        self.data = self.load_data()
        self.data_all = self.split_to_years(self.data)

        if self.matches is not None:
            self.data_all = self.get_match()

        self.X, self.y = self.get_x_y()
        self.X_lag, self.y_lag = self.get_lagged_data(self.X, self.y, self.lag)

        sample_item = self.__getitem__(0)
        self.input_shape = sample_item[0].shape[0]

    def encode_cats(self, batch):
        X, y = batch
        col_2_ins = X[0::len(self.columns)]
        col_3_ins = X[1::len(self.columns)]
        onehot_2 = self.onehot_2.transform(col_2_ins.reshape(-1, 1).astype(np.object_)).toarray()
        onehot_3 = self.onehot_3.transform(col_3_ins.reshape(-1, 1).astype(np.object_)).toarray()
        onehot_2 = np.argmax(onehot_2, axis=1)
        onehot_3 = np.argmax(onehot_3, axis=1)

        self.cat_2_size = len(self.onehot_2.categories_[0])
        self.cat_3_size = len(self.onehot_3.categories_[0])

        X = X.reshape(self.lag, -1)
        X = X[:, 2:]

        return onehot_2, onehot_3, X, y

    def get_match(self):
        train_matches = []
        for match in self.matches:
            try:
                store_match_train = self.data_all[np.where(self.data_all[:, 2] == match[0])[0]]
                single_match = store_match_train[np.where(store_match_train[:, 3] == match[1])[0]]
                if single_match.size != 0:
                    train_matches.append(single_match)
            except ValueError:
                print(f'[INFO] No match found for {match}, skipping...')
                continue
        return train_matches

    def get_x_y(self):
        X_all = []
        y_all = []
        for d in self.data_all:
            X_all.append(d[:, self.columns])
            y_all.append(d[:, -1])
        return X_all, y_all

    def get_lagged_data(self, X, y, lag):
        X_lagged = []
        y_lagged = []
        for n in range(len(X)):
            for i in range(X[n].shape[0] - (lag - 1) - 1):
                X_lagged.append(X[n][i:i + lag, :])
                y_lagged.append(y[n][i + lag])
        return np.array(X_lagged).reshape(len(X_lagged), -1), np.array(y_lagged)

    def load_data(self):
        return pd.read_csv(self.path)

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

    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __len__(self):
        return self.X_lag.shape[0]

    def __getitem__(self, idx):
        batch = self.X_lag[idx].astype(float), self.y_lag[idx].astype(float)
        emb_2, emb_3, X, y = self.encode_cats(batch)
        if self.normalize:
            X = self.normalize_data(X)
        return torch.LongTensor(emb_2), torch.LongTensor(emb_3), torch.Tensor(X), torch.Tensor([y])


def get_matches(path, encode=False):
    data = pd.read_csv(path)
    data = data.to_numpy()
    if encode:
        wh_encoder = OrdinalEncoder()
        whs = data[:, 1]
        whs_enc = wh_encoder.fit_transform(whs.reshape(-1, 1))

        prod_encoder = OrdinalEncoder()
        prods = data[:, 2]
        prods_enc = prod_encoder.fit_transform(prods.reshape(-1, 1))

        data[:, 1] = whs_enc.flatten()
        data[:, 2] = prods_enc.flatten()

    pairs = data[:, [1, 2]].astype(int)
    # get cartesian product for store-sku
    uq_pairs = np.unique(pairs, axis=0)
    # pick one pair for experiments
    return uq_pairs
