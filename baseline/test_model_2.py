from base_src import MLP_dataset, MLP_dataset_emb_2
from base_src import get_matches
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


MODEL_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/transfer/mlp_model_2.pth'
DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting-kernels-only/train.csv'

embedders = {'C2': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C2.pkl'},
             'C3': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C3.pkl'}}

OUT_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/transfer'
OUT_NAME = f'base_test_2'

out_dict = {}
matches = get_matches(DATA_PATH)

model = torch.load(MODEL_PATH).to('cuda')
model.eval()
loss = RMSELoss()

for m in tqdm(matches):
    val_data = MLP_dataset_emb_2(path=DATA_PATH, train=False, lag=15, get_quant=True, normalize=False,
                                 embedders=embedders, matches=m)

    samples_num = val_data.y_lag.shape[0]

    losses = []
    for i in range(samples_num):
        batch = val_data.__getitem__(i)
        batch = [b.unsqueeze(0).to('cuda') for b in batch]
        emb_2, emb_3, x, y = batch

        out = model(emb_2, emb_3, x)
        losses.append(loss(out, y).detach().cpu().numpy().item())

    mean_loss = np.mean(losses)
    out_dict[f'{m[0]}_{m[1]}'] = mean_loss

pickle.dump(out_dict, open(f'{OUT_PATH}/{OUT_NAME}.pkl', 'wb'))
