import numpy as np
import torch

class DataSet:
    def __init__(self, paths, recombine_data: list = [0.6, 0.2, 0.2]):
        self.path_train = paths['TRAIN']
        self.path_test = paths['TEST']
        self.name = self.path_train.split('/')[-2]
        self.recombine_data = recombine_data

        self.train = None
        self.val = None
        self.test = None
        self.header = None

        self.load_data()

    def load_data(self):
        train = None
        test = None

        # try header
        try:
            self.header = open(self.path_train, 'r').readline()
        except FileNotFoundError:
            self.header = None

        # try traning csv
        try:
            train = np.loadtxt(self.path_train, dtype=str, skiprows=1)
        except FileNotFoundError:
            train = None

        # try testing csv
        try:
            test = np.loadtxt(self.path_test, dtype=str, skiprows=1)
        except FileNotFoundError:
            test = None

        if self.recombine_data:
            self.recombine(train, test)


    def recombine(self, train, test):
        if test is None:
            data = train
        else:
            data = np.hstack([train, test])
        print(data.shape)
