import numpy as np
from torch.utils.data import DataLoader
from .mlp_dataset import MLP_dataset_cluster


class MatchBank:
    def __init__(self, init_matches: np.array, embedders: dict, threshold: float):
        self.embedders = embedders
        self.init_matches = init_matches

        self.matches_left = init_matches
        self.matches_used = []

        self.training_matches = None
        self.single_train_match = np.array([init_matches[0]])

    def test_phase(self, model, loss_fn):
        rmse_results = {}

        for m in self.matches_left:
            val_data = MLP_dataset_cluster(path='/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv',
                                           train=False, lag=15, get_quant=True, normalize=True,
                                           embedders=self.embedders, matches=np.array([m]))

            b_size = val_data.y_lag.shape[0] - 1
            val_dataloader = DataLoader(val_data, batch_size=b_size, shuffle=True, num_workers=15)

            emb_2, emb_3, X, flag = next(iter(val_dataloader))
            model_output = model(emb_2, emb_3, X)
            rmse_test = loss_fn(model_output, flag).detach().cpu().numpy().item()

