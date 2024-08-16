#%%
import torch
from src import Embedding_dataset
from src import Embedder, Embedder_double
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
COL = [2, 3]

data_train = Embedding_dataset(DATA_PATH, COL, True, out_path=OUT_PATH)
data_val = Embedding_dataset(DATA_PATH, COL, False, label_encoders=data_train.label_encoders)

train_dataloader = DataLoader(data_train, batch_size=BATCH, shuffle=True, num_workers=15)
val_dataloader = DataLoader(data_val, batch_size=BATCH, shuffle=False, num_workers=15)

# get model
model = Embedder_double(data_train.data_shape[0], data_train.data_shape[1], EMBEDDING_DIM).to(DEVICE)

# set loss
loss = RMSELoss()

# set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=False)

# set trainer
light_model = L_Net(model=model, loss_fn=loss, optimizer=optimizer, out_path=OUT_PATH)
lightning_trainer = L.Trainer(accelerator=DEVICE, max_epochs=EPOCHS, limit_train_batches=1000, limit_val_batches=500,
                              check_val_every_n_epoch=1, log_every_n_steps=20, enable_progress_bar=True)

# train
lightning_trainer.fit(model=light_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
print('[INFO] DONE')
#%%
x1, x2, lbl = next(iter(train_dataloader))
model = torch.load('/home/mateusz/Desktop/Demand-Forecast/embedding_models/model.pth')
embedder = torch.load('/home/mateusz/Desktop/Demand-Forecast/embedding_models/embedder_c3.pth')

model_out = model(x1, x2)
embedder_out = embedder(x2)

print(model_out)
print(embedder_out)
#%%
import torch
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

COL = 2

onehot_embedder = pickle.load(open(f'/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C{COL}.pkl', 'rb'))
categories = onehot_embedder.categories_
onehots = onehot_embedder.transform(np.array(categories).T).toarray()
embedder_inputs = torch.LongTensor(np.argmax(onehots, axis=1))

embedder = torch.load(f'/home/mateusz/Desktop/Demand-Forecast/embedding_models/embedder_c{COL}.pth')

out_embeddings = embedder(embedder_inputs).detach().numpy()
similarities = cosine_similarity(out_embeddings)
similarities = np.where(np.eye(similarities.shape[0]) == 1, np.nan, similarities)

# get most similar items
most_similar = np.argwhere(similarities == np.nanmax(similarities))[0]
# get the least similar item to first of best from most similar pair
least_similar = np.argwhere(similarities == np.nanmin(similarities))[0]
min_max_sim_cat = np.array(categories).flatten()[np.concatenate([most_similar, least_similar])]


sns.heatmap(similarities)
plt.show()
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

similar_skus = [320485, 398721, 216418, 320485]
similar_stores = [9132, 9425, 9789, 9872]

path = 'DS/demand-forecasting/train.csv'
data = pd.read_csv(path)
skus = data.sku_id.to_numpy()
stores = data.store_id.to_numpy()
counts = data.units_sold.to_numpy()
weeks = data.week.to_numpy()

stores_counts = {}
skus_counts = {}

for i in range(len(similar_skus)):
    skus_counts[similar_skus[i]] = {}
    skus_counts[similar_skus[i]]['count'] = counts[np.where(skus == similar_skus[i])]
    skus_counts[similar_skus[i]]['week'] = weeks[np.where(skus == similar_skus[i])]

for i in range(len(similar_skus)):
    stores_counts[similar_stores[i]] = {}
    stores_counts[similar_stores[i]]['count'] = counts[np.where(stores == similar_stores[i])]
    stores_counts[similar_stores[i]]['week'] = weeks[np.where(stores == similar_stores[i])]

print(f'SKUS: \nsimilar 1 count = {np.sum(skus_counts[similar_skus[0]]["count"])}'
      f'\nsimilar 2 count = {np.sum(skus_counts[similar_skus[1]]["count"])}'
      f'\ndissimilar 1 count = {np.sum(skus_counts[similar_skus[2]]["count"])}'
      f'\ndissimilar 2 count = {np.sum(skus_counts[similar_skus[3]]["count"])}')

print(f'STORES: \nsimilar 1 count = {np.sum(stores_counts[similar_stores[0]]["count"])}'
      f'\nsimilar 2 count = {np.sum(stores_counts[similar_stores[1]]["count"])}'
      f'\ndissimilar 1 count = {np.sum(stores_counts[similar_stores[2]]["count"])}'
      f'\ndissimilar 2 count = {np.sum(stores_counts[similar_stores[3]]["count"])}')

alpha = 0.65
fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(15, 7), layout='constrained')
axs[0].plot(stores_counts[similar_stores[0]]['week'], stores_counts[similar_stores[0]]['count'],
            label='Similar 1', alpha=alpha)
axs[0].plot(stores_counts[similar_stores[1]]['week'], stores_counts[similar_stores[1]]['count'],
            label='Similar 2', alpha=alpha)
axs[0].plot(stores_counts[similar_stores[2]]['week'], stores_counts[similar_stores[2]]['count'],
            label='Dissimilar 1', alpha=alpha)
axs[0].plot(stores_counts[similar_stores[3]]['week'], stores_counts[similar_stores[3]]['count'],
            label='Dissimilar 2', alpha=alpha)
axs[1].plot(skus_counts[similar_skus[0]]['week'], skus_counts[similar_skus[0]]['count'],
            label='Similar 1', alpha=alpha)
axs[1].plot(skus_counts[similar_skus[1]]['week'], skus_counts[similar_skus[1]]['count'],
            label='Similar 2', alpha=alpha)
axs[1].plot(skus_counts[similar_skus[2]]['week'], skus_counts[similar_skus[2]]['count'],
            label='Dissimilar 1', alpha=alpha)
axs[1].plot(skus_counts[similar_skus[3]]['week'], skus_counts[similar_skus[3]]['count'],
            label='Dissimilar 2', alpha=alpha)

axs[0].set_title('Similar stores')
axs[1].set_title('Similar skus')
fig.suptitle('Store/SKU similarities')
axs[0].legend()
axs[0].grid()
axs[0].tick_params(axis='x', rotation=45, labelsize=5)
axs[1].legend()
axs[1].grid()
axs[1].tick_params(axis='x', rotation=45, labelsize=5)
plt.show()
