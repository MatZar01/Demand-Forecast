from base_src import MLP_dataset, MLP_dataset_emb, MLP_dataset_cluster
from base_src import get_matches
from base_src import MLP, MLP_emb, tl_model, MLP_emb_tl
from base_src import L_Net, L_Net_TL
from base_src import MatchBank
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
LR = 0.0001
EPOCHS = 20
QUANT = True
EMBED = True
NORMALIZE = True
THRESHOLD = 20

MAX_MODELS = 1500
MIN_DATA_PER_MODEL = 10

DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv'
embedders = {'C2': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C2.pkl'},
             'C3': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C3.pkl'}}

matches = get_matches(DATA_PATH)
match_bank = MatchBank(init_matches=matches, embedders=embedders, threshold=THRESHOLD)

if not EMBED:
    embedders = None

SAVE_MODEL = False

OUT_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering'

out_dict = {}
model_num = 0

# training method:
# phase 1 - find good match and train initially on single match
# phase 2 - test model on the rest of matches
# phase 3 - train the model on the matches group
# repeat for the leftover matches
# phase 4 - train last model for crumbs

for i in range(MAX_MODELS):
    # break if no more data and update threshold
    if len(match_bank.matches_left) < MIN_DATA_PER_MODEL:
        break
    if match_bank.threshold > 36:
        break

    # PHASE 1
    while True:
        try:
            train_data = MLP_dataset_cluster(path=DATA_PATH, train=True, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                             embedders=embedders, matches=match_bank.single_train_match)
            val_data = MLP_dataset_cluster(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                           embedders=embedders, matches=match_bank.single_train_match)
        except ValueError:
            print('[INFO] No match found, skipping...')
            match_bank.change_idx()

        train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=0)

        model = MLP_emb_tl(input_dim=train_data.input_shape, cat_2_size=train_data.cat_2_size, cat_3_size=train_data.cat_3_size, embedding_size=5)

        # set loss
        loss = RMSELoss()

        # set optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)

        # set trainer
        light_model = L_Net(model=model, loss_fn=loss, optimizer=optimizer, out_path=OUT_PATH,
                            save_model=SAVE_MODEL)

        lightning_trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, limit_train_batches=4000,
                                      limit_val_batches=500,
                                      check_val_every_n_epoch=1, log_every_n_steps=20, enable_progress_bar=True)

        # train
        lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        if light_model.out_dict['rmse_test'] < THRESHOLD:
            print(f'[INFO] Found good match {match_bank.single_train_match} with RMSE: {light_model.out_dict["rmse_test"]}')
            break
        else:
            match_bank.change_idx()

    # PHASE 2 (skip if low on matches)
    match_bank.test_phase(model=model, loss_fn=loss)
    match_bank.assess_test()
    if len(match_bank.matches_used) < MIN_DATA_PER_MODEL:
        match_bank.change_idx()
        match_bank.swap_matches()
        continue

    # PHASE 3
    train_data = MLP_dataset_cluster(path=DATA_PATH, train=True, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                     embedders=embedders, matches=match_bank.matches_used)
    val_data = MLP_dataset_cluster(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                   embedders=embedders, matches=match_bank.matches_used)

    train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=0)

    model = MLP_emb_tl(input_dim=train_data.input_shape, cat_2_size=train_data.cat_2_size, cat_3_size=train_data.cat_3_size, embedding_size=5)

    # set loss
    loss = RMSELoss()

    # set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)

    # set trainer
    light_model = L_Net(model=model, loss_fn=loss, optimizer=optimizer, out_path=OUT_PATH,
                        save_model=SAVE_MODEL)

    lightning_trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, limit_train_batches=4000,
                                  limit_val_batches=500,
                                  check_val_every_n_epoch=1, log_every_n_steps=20, enable_progress_bar=True)

    # train
    lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    out_dict[model_num] = {}
    out_dict[model_num]['matches'] = match_bank.matches_used
    out_dict[model_num]['rmse'] = light_model.out_dict['rmse_test']
    model_num += 1

    # update match bank
    match_bank.matches_used = []
    match_bank.single_train_match = [match_bank.matches_left[0]]

# PHASE 4 (PHASE 3 repeated for leftovers)
match_bank.matches_used = list(match_bank.matches_left)

train_data = MLP_dataset_cluster(path=DATA_PATH, train=True, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                 embedders=embedders, matches=match_bank.matches_used)
val_data = MLP_dataset_cluster(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                               embedders=embedders, matches=match_bank.matches_used)

train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=0)

model = MLP_emb(input_dim=train_data.input_shape, cat_2_size=train_data.cat_2_size, cat_3_size=train_data.cat_3_size, embedding_size=5)

# set loss
loss = RMSELoss()

# set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)

# set trainer
light_model = L_Net(model=model, loss_fn=loss, optimizer=optimizer, out_path=OUT_PATH,
                    save_model=SAVE_MODEL)

lightning_trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, limit_train_batches=4000,
                              limit_val_batches=500,
                              check_val_every_n_epoch=1, log_every_n_steps=20, enable_progress_bar=True)

# train
lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

out_dict[model_num] = {}
out_dict[model_num]['matches'] = match_bank.matches_used
out_dict[model_num]['rmse'] = light_model.out_dict['rmse_test']

pickle.dump(out_dict, open(f'{OUT_PATH}/model_out_grouper.pkl', 'wb'))

