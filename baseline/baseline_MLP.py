from base_src import MLP_dataset
from base_src import get_matches
from base_src import MLP
from base_src import L_Net
import torch
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
LAG = 10
WEIGHT_DECAY = 0.004
LR = 0.001
EPOCHS = 100

DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv'
embedders = {'C2': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C2.pkl',
                    'cat2vec': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/embedder_c2.pth'},
             'C3': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C3.pkl',
                    'cat2vec': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/embedder_c3.pth'}}


matches = get_matches(DATA_PATH)
train_data = MLP_dataset(path=DATA_PATH, train=True, lag=LAG, get_quant=True, normalize=True,
                         embedders=None, matches=matches[0])
val_data = MLP_dataset(path=DATA_PATH, train=False, lag=LAG, get_quant=True, normalize=True,
                       embedders=None, matches=matches[0])

train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=15)
val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=True, num_workers=15)

model = MLP(input_dim=train_data.input_shape)

# set loss
loss = RMSELoss()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)

# set trainer
light_model = L_Net(model=model, loss_fn=loss, optimizer=optimizer)
lightning_trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, limit_train_batches=1000, limit_val_batches=500,
                              check_val_every_n_epoch=1, log_every_n_steps=20, enable_progress_bar=True)

# train
lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
print('[INFO] DONE')
