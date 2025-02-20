from base_src import MLP_dataset, MLP_dataset_emb
from base_src import get_matches
from base_src import MLP_emb, tl_model
from base_src import L_Net, L_Net_TL
import torch
from torch.utils.data import DataLoader
import lightning as L
import numpy as np
import pickle


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


DEVICE = 'cuda'
BATCH = 8
LAG = 15
WEIGHT_DECAY = 0.004
LR = 0.001
EPOCHS = 40
QUANT = True
EMBED = True
NORMALIZE = True
MATCHES_ONLY = True

DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv'
embedders = {'C2': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C2.pkl'},
             'C3': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C3.pkl'}}

MODEL_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/transfer/mlp_model.pth'

if not EMBED:
    embedders = None

SAVE_MODEL = False

OUT_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/transfer'
OUT_NAME = f'L_{LAG}_Q_{QUANT}_EM_{EMBED}_FT'

base_model = torch.load(MODEL_PATH)
base_model.embedder_2.requires_grad_(False)
base_model.embedder_3.requires_grad_(False)
base_model.model.requires_grad_(False)
base_model.clf = torch.nn.Identity()

out_dict = {}
matches = get_matches(DATA_PATH)

if not MATCHES_ONLY:
    matches = [None]

for m in matches:
    try:
        train_data = MLP_dataset_emb(path=DATA_PATH, train=True, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                 embedders=embedders, matches=m)
        val_data = MLP_dataset_emb(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                               embedders=embedders, matches=m)

        train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=15)
        val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=True, num_workers=15)

        model = tl_model()

        # set loss
        loss = RMSELoss()

        # set optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)

        # set trainer
        light_model = L_Net_TL(model=model, base_model=base_model, loss_fn=loss, optimizer=optimizer, out_path=OUT_PATH, save_model=SAVE_MODEL)
        lightning_trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, limit_train_batches=1000, limit_val_batches=500,
                                      check_val_every_n_epoch=1, log_every_n_steps=20, enable_progress_bar=True)

        # train
        lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        out_dict[f'{m[0]}_{m[1]}'] = light_model.out_dict

        print(f'[INFO] DONE {m}')
    except:
        out_dict[f'{m[0]}_{m[1]}'] = {'rmse_train': np.nan, 'rmse_test': np.nan}
        print(f'[INFO] {m} -- no data!')

pickle.dump(out_dict, open(f'{OUT_PATH}/{OUT_NAME}.pkl', 'wb'))
