from base_src import MLP_dataset_emb
from base_src import get_matches
from base_src import MLP, MLP_emb, MLP_emb_tl
from base_src import L_Net
import torch
from torch.utils.data import DataLoader
import lightning as L
import numpy as np
import pickle
import traceback

import os
import json
import argparse

# Create the parser
parser = argparse.ArgumentParser()
# Add a string argument
parser.add_argument('-d','--dir', type=str, help="file", default='../DS/NZ_energy_latest_LAG_0/')
# Parse the arguments
args = parser.parse_args()

# Reading config from JSON file
config = None
with open(os.path.join(args.dir, 'config.json') , 'r') as file:
    config = json.load(file)


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


# DEVICE = 'cuda'
DEVICE = 'cpu'
BATCH = 8
LAG = 0
WEIGHT_DECAY = 0.004
LR = 0.001
EPOCHS = 200
QUANT = None # for NZ energy
# QUANT = 8 # previous target feature, for demand forcasting
EMBED = True
NORMALIZE = True
MATCHES_ONLY = False

OUT_PATH = args.dir
DATA_PATH = f"{os.path.join(args.dir, config['data'])}"
embedders = {'C2': {'onehot': f"{os.path.join(args.dir, config['C2'])}"},
             'C3': {'onehot': f"{os.path.join(args.dir, config['C3'])}"}}

if not EMBED:
    embedders = None


# OUT_PATH = '../baseline/results_mlp/transfer'
OUT_NAME = f'L_{LAG}_Q_{QUANT}_EM_{EMBED}'
SAVE_MODEL = True

out_dict = {}
matches = get_matches(DATA_PATH)

if not MATCHES_ONLY:
    matches = [None]

if __name__ == '__main__':
    # freeze_support()
    for m in matches:
        try:
            train_data = MLP_dataset_emb(path=DATA_PATH, train=True, lag=LAG, quant_feature=QUANT, normalize=NORMALIZE,
                                         embedders=embedders, matches=m, data_split='train', columns=[0,1,2,3,4,5,6,7,8,9,10,11], dataset='NZ energy')
            val_data = MLP_dataset_emb(path=DATA_PATH, train=False, lag=LAG, quant_feature=QUANT, normalize=NORMALIZE,
                                        embedders=embedders, matches=m, data_split='val', columns=[0,1,2,3,4,5,6,7,8,9,10,11], dataset='NZ energy')

            train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=1)
            val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=True, num_workers=1)

            model = MLP_emb_tl(input_dim=train_data.input_shape, cat_2_size=train_data.cat_2_size, cat_3_size=train_data.cat_3_size, embedding_size=5, lag=LAG)

            # set loss
            loss = RMSELoss()

            # set optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)

            # set trainer
            light_model = L_Net(model=model, loss_fn=loss, optimizer=optimizer, out_path=OUT_PATH, save_model=SAVE_MODEL)
            lightning_trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, limit_train_batches=1000, limit_val_batches=500,
                                          check_val_every_n_epoch=1, log_every_n_steps=20, enable_progress_bar=True)

            # train
            lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

            if None in matches:
                out_dict = light_model.out_dict
            else:
                out_dict[f'{m[0]}_{m[1]}'] = light_model.out_dict

            print(f'[INFO] DONE {m}')
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()  # Prints the full traceback, including the line number
            # out_dict[f'{m[0]}_{m[1]}'] = {'rmse_train': np.nan, 'rmse_test': np.nan}
            print(f'[INFO] {m} -- no data!')

    pickle.dump(out_dict, open(f'{OUT_PATH}/{OUT_NAME}.pkl', 'wb'))
