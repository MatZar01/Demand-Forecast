import pickle
from base_src import MLP_dataset, MLP_dataset_emb, MLP_dataset_cluster
from base_src import MLP, MLP_emb, tl_model, MLP_emb_tl, Conv_1D, MLP_emb_pool
from base_src import L_Net, L_Net_TL
import torch
from torch.utils.data import DataLoader
import lightning as L
from torchmetrics import MeanSquaredLogError, SymmetricMeanAbsolutePercentageError


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


DEVICE = 'cuda'
BATCH = 2
LAG = 15
WEIGHT_DECAY = 0.004
LR = 0.001
EPOCHS = 25
QUANT = True
EMBED = True
NORMALIZE = True
OUT_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/model_grouping_refinement.pkl'

DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv'
embedders = {'C2': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C2.pkl'},
             'C3': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C3.pkl'}}

if not EMBED:
    embedders = None

SAVE_MODEL = False

data_grouping = pickle.load(open('/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/model_out_grouper_1.pkl', 'rb'))
model_num = len(data_grouping.keys())
model_iter = 0

out_dict = {}

for key in data_grouping.keys():

    matches = data_grouping[key]['matches']
    goal_rmse = data_grouping[key]['rmse']

    train_data = MLP_dataset_cluster(path=DATA_PATH, train=True, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                                 embedders=embedders, matches=matches)
    val_data = MLP_dataset_cluster(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                   embedders=embedders, matches=matches)

    train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=False, num_workers=0)

    model = MLP_emb_tl(input_dim=train_data.input_shape, cat_2_size=train_data.cat_2_size, cat_3_size=train_data.cat_3_size, embedding_size=5)

    # set loss and test fns
    test_fn = RMSELoss()
    loss = MeanSquaredLogError()

    # set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)

    # set trainer
    light_model = L_Net(model=model, loss_fn=loss, test_fn=test_fn, optimizer=optimizer, out_path=OUT_PATH,
                        save_model=SAVE_MODEL)

    lightning_trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, limit_train_batches=4000,
                                  limit_val_batches=500, logger=False, enable_checkpointing=False,
                                  check_val_every_n_epoch=1, log_every_n_steps=20, enable_progress_bar=True)

    # train
    lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    out_dict[key] = {}
    out_dict[key]['matches'] = data_grouping[key]['matches']
    out_dict[key]['old_rmse'] = data_grouping[key]['rmse']
    out_dict[key]['new_rmse'] = light_model.out_dict['rmse_test']
    model_iter += 1
    print(f'[INFO] {model_iter}/{model_num} models trained')

pickle.dump(out_dict, open(OUT_PATH, 'wb'))
