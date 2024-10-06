import pickle
import numpy as np
from base_src import MLP_dataset, MLP_dataset_emb, MLP_dataset_cluster
from base_src import get_matches
from base_src import MLP, MLP_emb, tl_model, MLP_emb_tl, Conv_1D, MLP_emb_pool
from base_src import L_Net, L_Net_TL
from base_src import MatchBank
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as L


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
EPOCHS = 30
QUANT = True
EMBED = True
NORMALIZE = True


DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv'
embedders = {'C2': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C2.pkl'},
             'C3': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C3.pkl'}}
data_grouping = pickle.load(open('/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/model_out_grouper_1.pkl', 'rb'))

matches = data_grouping[18]['matches']
goal_rmse = data_grouping[18]['rmse']

if not EMBED:
    embedders = None

SAVE_MODEL = False

OUT_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering'

train_data = MLP_dataset_cluster(path=DATA_PATH, train=True, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                             embedders=embedders, matches=matches)
val_data = MLP_dataset_cluster(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                               embedders=embedders, matches=matches)

train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=0)

model = MLP_emb_pool(input_dim=train_data.input_shape, cat_2_size=train_data.cat_2_size, cat_3_size=train_data.cat_3_size, embedding_size=5)

# set loss and test fns
test_fn = RMSELoss()
#loss = nn.SmoothL1Loss()
loss = RMSELoss()

# set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
#optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, nesterov=True, momentum=0.4)

# set trainer
light_model = L_Net(model=model, loss_fn=loss, test_fn=test_fn, optimizer=optimizer, out_path=OUT_PATH,
                    save_model=SAVE_MODEL)

lightning_trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, limit_train_batches=4000,
                              limit_val_batches=500, logger=False, enable_checkpointing=False,
                              check_val_every_n_epoch=1, log_every_n_steps=20, enable_progress_bar=True)

# train
lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
