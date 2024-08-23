from base_src import MLP_dataset
from base_src import get_matches
from base_src import MLP
from base_src import L_Net
import torch
from torch.utils.data import DataLoader
import lightning as L
import numpy as np
import pickle
from tqdm import tqdm


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


DEVICE = 'cuda'
LAG = 15
QUANT = True
EMBED = True
NORMALIZE = True
MATCHES_ONLY = True

DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv'
embedders = {'C2': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C2.pkl',
                    'cat2vec': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/embedder_c2.pth'},
             'C3': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C3.pkl',
                    'cat2vec': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/embedder_c3.pth'}}

model = torch.load('/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/mlp_model.pth')
loss_fn = RMSELoss()

if not EMBED:
    embedders = None


OUT_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/whole_model_test.pkl'
OUT_NAME = f'L_{LAG}_Q_{QUANT}_EM_{EMBED}'

out_dict = {}
matches = get_matches(DATA_PATH)

if not MATCHES_ONLY:
    matches = [None]

#matches = [matches[0]]
for m in tqdm(matches):
    try:
        val_data = MLP_dataset(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                               embedders=embedders, matches=m)

        b_size = val_data.y_lag.shape[0] - 1

        val_dataloader = DataLoader(val_data, batch_size=b_size, shuffle=True, num_workers=15)

        input, flag = next(iter(val_dataloader))
        model_output = model(input)
        rmse_test = loss_fn(model_output, flag).detach().cpu().numpy().item()

        out_dict[f'{m[0]}_{m[1]}'] = {'rmse_train': np.nan, 'rmse_test': rmse_test}

    except:
        out_dict[f'{m[0]}_{m[1]}'] = {'rmse_train': np.nan, 'rmse_test': np.nan}

pickle.dump(out_dict, open(OUT_PATH, 'wb'))
