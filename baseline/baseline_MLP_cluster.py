from base_src import MLP_dataset, MLP_dataset_emb, MLP_dataset_cluster
from base_src import get_matches
from base_src import MLP, MLP_emb, tl_model
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
LR = 0.0001
EPOCHS = 100
QUANT = True
EMBED = True
NORMALIZE = True
MATCHES_ONLY = True

DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv'
embedders = {'C2': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C2.pkl'},
             'C3': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C3.pkl'}}

MATCHES_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/emb_assignments.pkl'
assignments = pickle.load(open(MATCHES_PATH, 'rb'))

if not EMBED:
    embedders = None

SAVE_MODEL = False

OUT_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering'

out_dict = {25: [], 50: [], 100: []}

model_groups = [25, 50, 100]

for group in model_groups:
    matches_per_model = assignments[group]
    model_nums = list(range(group))
    for num in model_nums:
        try:
            matches = matches_per_model[num]
            matches = np.array([[int(a.split('_')[0]), int(a.split('_')[1])] for a in matches])

            train_data = MLP_dataset_cluster(path=DATA_PATH, train=True, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                             embedders=embedders, matches=matches)
            val_data = MLP_dataset_cluster(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                                           embedders=embedders, matches=matches)

            train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=15)
            val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=True, num_workers=15)

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

            out_dict[group].append(light_model.out_dict['rmse_test'])

        except:
            print(f'[INFO] model from {group}, No. {num} failed')

pickle.dump(out_dict, open(f'{OUT_PATH}/model_out.pkl', 'wb'))

print(f'Models mean rmse:\n'
      f'Group 25: {np.mean(out_dict[25])}\n'
      f'Group 50: {np.mean(out_dict[50])}\n'
      f'Group 100: {np.mean(out_dict[100])}')