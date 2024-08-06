#%%
import torch
from src import Embedding_dataset
from src import Embedder
from src import L_Net
from torch.utils.data import DataLoader
import importlib
import lightning as L


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


DATA_PATH = 'DS/demand-forecasting/train.csv'
OUT_PATH = 'embedding_models'
EMBEDDING_DIM = 5
LR = 0.001
WEIGHT_DECAY = 0.004
DEVICE = 'cuda'
EPOCHS = 10
BATCH = 8
COL = 2

data_train = Embedding_dataset(DATA_PATH, COL, True)
data_val = Embedding_dataset(DATA_PATH, COL, False, label_encoder=data_train.label_encoder)

train_dataloader = DataLoader(data_train, batch_size=BATCH, shuffle=True, num_workers=15)
val_dataloader = DataLoader(data_val, batch_size=BATCH, shuffle=False, num_workers=15)

# get model
model = Embedder(data_train.data_shape, EMBEDDING_DIM).to(DEVICE)

# set loss
#loss = torch.nn.MSELoss()
#loss = torch.nn.L1Loss()
loss = RMSELoss()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)
#optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# set trainer
light_model = L_Net(model=model, loss_fn=loss, optimizer=optimizer, out_path=OUT_PATH, col=COL)
lightning_trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, limit_train_batches=1000, limit_val_batches=500,
                              check_val_every_n_epoch=1, log_every_n_steps=20, enable_progress_bar=True)

# train
lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
print('[INFO] DONE')
#%%