import numpy as np
from torch.utils.data import DataLoader
from .mlp_dataset import MLP_dataset_cluster
from tqdm import tqdm


class MatchBank:
    def __init__(self, init_matches: np.array, embedders: dict, threshold: float):
        self.embedders = embedders
        self.init_matches = list(init_matches)

        self.matches_left = list(init_matches)
        self.matches_used = []

        self.current_idx = 0
        self.single_train_match = [self.init_matches[self.current_idx]]

        self.test_results = None
        self.threshold = threshold

    def test_phase(self, model, loss_fn):
        print('[INFO] Testing...')
        rmse_results = {}

        for m in tqdm(self.matches_left):
            try:
                val_data = MLP_dataset_cluster(path='/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv',
                                               train=False, lag=15, get_quant=True, normalize=True,
                                               embedders=self.embedders, matches=np.array([m]))
            except ValueError:
                continue

            b_size = val_data.y_lag.shape[0] - 1
            val_dataloader = DataLoader(val_data, batch_size=b_size, shuffle=True, num_workers=0)

            emb_2, emb_3, X, flag = next(iter(val_dataloader))
            model_output = model(emb_2, emb_3, X)
            rmse_test = loss_fn(model_output, flag).detach().cpu().numpy().item()
            rmse_results[f'{m[0]}_{m[1]}'] = rmse_test

        print('[INFO] Tests ended')
        self.test_results = rmse_results

    def assess_test(self):
        print('[INFO] Assessing...')
        matches_used = []
        matches_left = []
        for key in tqdm(self.test_results.keys()):
            match = np.array([int(key.split('_')[0]), int(key.split('_')[1])])
            rmse = self.test_results[key]
            if rmse < self.threshold * 1.6:
                matches_used.append(match)
            else:
                matches_left.append(match)

        self.matches_used = matches_used
        self.matches_left = matches_left

    def change_idx(self):
        self.current_idx += 1
        try:
            self.single_train_match = [self.matches_left[self.current_idx]]
        except IndexError:
            self.single_train_match = [self.matches_left[0]]
            self.threshold += 5
            self.current_idx = 0

    def swap_matches(self):
        self.matches_left = self.matches_left + self.matches_used
        self.matches_used = []
