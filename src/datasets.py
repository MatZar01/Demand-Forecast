import numpy as np
import pandas as pd


class DataSet:
    def __init__(self, paths, year_split: bool = True, train_split: float = None):
        self.path_train = paths['TRAIN']
        self.path_test = paths['TEST']
        self.name = self.path_train.split('/')[-2]
        self.year_split = year_split
        self.split = train_split

        self.train_all = None
        self.val = None
        self.test = None
        self.train_y1 = None
        self.train_y2 = None
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

        if self.year_split:
            self.split_to_years(df_train)
            print(f'[INFO] Data split to years:\ntrain 1st year: {self.train_y1.shape[0]} pts\n'
                  f'train 2nd year: {self.train_y2.shape[0]} pts\n'
                  f'val (3rd year): {self.val.shape[0]} pts\n'
                  f'test: {self.test.shape[0]} pts')
        elif self.split is not None:
            self.simple_split(df_train)
            print(f'[INFO] Data with no year split:\ntrain: {self.train_all.shape[0]} pts\n'
                  f'val: {self.val.shape[0]} pts\n'
                  f'test: {self.test.shape[0]} pts')
        else:
            self.train_all = df_train.to_numpy()
            print(f'[INFO] Data with no split:\ntrain: {self.train_all.shape[0]} pts\n'
                  f'test: {self.test.shape[0]} pts')

        print(f'Header: {self.header}')

    def split_to_years(self, train):
        data = train.to_numpy()
        years = np.array([int(x.split('/')[-1]) for x in data[:, 1]])
        years = years - np.min(years)

        self.train_y1 = data[np.where(years == 0)]
        self.train_y2 = data[np.where(years == 1)]
        self.val = data[np.where(years == 2)]
        self.train_all = np.vstack([self.train_y1, self.train_y2])

    def simple_split(self, train):
        data = train.to_numpy()
        split_index = int(data.shape[0] * self.split)
        self.train_all = data[:split_index, :]
        self.val = data[split_index:, :]


class DataSet_2:
    def __init__(self, paths, year_split: bool = True, train_split: float = None):
        self.path_train = paths['TRAIN']
        self.path_test = paths['TEST']
        self.name = self.path_train.split('/')[-2]
        self.year_split = year_split
        self.split = train_split

        self.train_all = None
        self.val = None
        self.test = None
        self.train = None
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

        if self.year_split:
            self.split_to_years(df_train)
            print(f'[INFO] Data split to years:\ntrain: {self.train.shape[0]} pts\n'
                  f'val: {self.val.shape[0]} pts\n'
                  f'test: {self.test.shape[0]} pts')
        elif self.split is not None:
            self.simple_split(df_train)
            print(f'[INFO] Data with no year split:\ntrain: {self.train_all.shape[0]} pts\n'
                  f'val: {self.val.shape[0]} pts\n'
                  f'test: {self.test.shape[0]} pts')
        else:
            self.train_all = df_train.to_numpy()
            print(f'[INFO] Data with no split:\ntrain: {self.train_all.shape[0]} pts\n'
                  f'test: {self.test.shape[0]} pts')

        print(f'Header: {self.header}')

    def split_to_years(self, train):
        data = train.to_numpy()
        years = np.array([int(x.split('-')[-0]) for x in data[:, 0]])

        self.train = data[np.where(years < 2016)]
        self.val = data[np.where(years >= 2016)]
        self.train_all = self.train

    def simple_split(self, train):
        data = train.to_numpy()
        split_index = int(data.shape[0] * self.split)
        self.train_all = data[:split_index, :]
        self.val = data[split_index:, :]


class DataSet_3:
    def __init__(self, paths, year_split: bool = True, train_split: float = None):
        self.path_train = paths['TRAIN']
        self.path_test = paths['TEST']
        self.name = self.path_train.split('/')[-2]
        self.year_split = year_split
        self.split = train_split

        self.train_all = None
        self.val = None
        self.test = None
        self.train = None
        self.header = None
        self.header_test = None

        self.load_data()

    def load_data(self):
        # try training data
        try:
            df_train = pd.read_csv(self.path_train)
            df_train = df_train.drop(df_train.loc[df_train.isnull().any(axis=1)].index)
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

        if self.year_split:
            self.split_to_years(df_train)
            print(f'[INFO] Data split to years:\ntrain: {self.train.shape[0]} pts\n'
                  f'val: {self.val.shape[0]} pts\n'
                  f'test: {self.test.shape[0]} pts')
        elif self.split is not None:
            self.simple_split(df_train)
            print(f'[INFO] Data with no year split:\ntrain: {self.train_all.shape[0]} pts\n'
                  f'val: {self.val.shape[0]} pts\n'
                  f'test: {self.test.shape[0]} pts')
        else:
            self.train_all = df_train.to_numpy()
            print(f'[INFO] Data with no split:\ntrain: {self.train_all.shape[0]} pts\n'
                  f'test: {self.test.shape[0]} pts')

        print(f'Header: {self.header}')

    def split_to_years(self, train):
        data = train.to_numpy()
        years = np.array([int(x.split('/')[0]) for x in data[:, 3]])

        self.train = data[np.where(years < 2015)]
        self.val = data[np.where(years >= 2015)]
        self.train_all = self.train

    def simple_split(self, train):
        data = train.to_numpy()
        split_index = int(data.shape[0] * self.split)
        self.train_all = data[:split_index, :]
        self.val = data[split_index:, :]
