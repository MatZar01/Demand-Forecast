import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path_mlp = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/L_15_Q_False_EM_False.pkl'

def get_data(path):
    data = pkl.load(open(path, 'rb'))
    stores = []
    skus = []
    rmses = []

    for key in data.keys():
        run = data[key]
        stores.append(key.split('_')[0])
        skus.append(key.split('_')[1])
        rmses.append(run['rmse_test'])

    stores_strip = list(set(stores))
    skus_strip = list(set(skus))
    return stores, stores_strip, skus, skus_strip, rmses, data

def get_out_matrix(data):
    out_matrix = np.zeros((len(data[1]), len(data[3])))
    for i in range(len(data[1])):
        for k in range(len(data[3])):
            out_matrix[i][k] = data[-1][f'{data[1][i]}_{data[3][k]}']['rmse_test']
    return out_matrix


def show_matrix(matrix, data):
    matrix[np.argwhere(matrix == np.inf)] = np.nan
    dims = (4.5, 8.27)
    plt.rcParams.update({'font.size': .81})
    sns.set(font_scale=.71)
    fig, ax = plt.subplots(figsize=dims)
    ax = sns.heatmap(matrix, xticklabels=data[3], yticklabels=data[1], square=True)
    ax.set_xlabel('SKU id')
    ax.set_ylabel('Store id')
    plt.title(f'RMSE for store-sku match, MEAN: {np.mean(matrix[np.isfinite(matrix)])}')
    plt.tight_layout()
    plt.show()


data_mlp = get_data(path_mlp)
out_mlp = get_out_matrix(data_mlp)
show_matrix(out_mlp, data_mlp)
#%%
path_mlp = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/L_15_Q_False_EM_True.pkl'
data_mlp = get_data(path_mlp)
out_mlp = get_out_matrix(data_mlp)
show_matrix(out_mlp, data_mlp)
#%%
path_mlp = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/L_15_Q_True_EM_False.pkl'
data_mlp = get_data(path_mlp)
out_mlp = get_out_matrix(data_mlp)
show_matrix(out_mlp, data_mlp)
#%%
path_mlp = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/L_15_Q_True_EM_True.pkl'
data_mlp = get_data(path_mlp)
out_mlp = get_out_matrix(data_mlp)
show_matrix(out_mlp, data_mlp)
#%%
### WHOLE MODEL - NO FINETUNING
path_mlp = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/whole_model_test.pkl'
data_mlp = get_data(path_mlp)
out_mlp = get_out_matrix(data_mlp)
show_matrix(out_mlp, data_mlp)
#%%
### WHOLE MODEL - FINETUNING
path_mlp = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/finetuning/L_15_Q_True_EM_True_FT.pkl'
data_mlp = get_data(path_mlp)
out_mlp = get_out_matrix(data_mlp)
show_matrix(out_mlp, data_mlp)
#%%
### WHOLE MODEL - EXTENDED EMBEDDING
path_mlp = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/embedding/whole_model_test.pkl'
data_mlp = get_data(path_mlp)
out_mlp = get_out_matrix(data_mlp)
show_matrix(out_mlp, data_mlp)
#%%
### ARIMA
path = '/home/mateusz/Desktop/Demand-Forecast/baseline/results/ARIMA_2024-7-16-4:24:10.pkl'
data = pkl.load(open(path, 'rb'))

stores = []
skus = []
rmses = []
preds = []
gts = []

for key in data.keys():
    run = data[key]
    stores.append(key.split('_')[0])
    skus.append(key.split('_')[1])
    rmses.append(run['rmse'])
    preds.append(run['preds'])
    gts.append(run['gt'])

stores_strip = list(set(stores))
skus_strip = list(set(skus))

out_matrix = np.zeros((len(stores_strip), len(skus_strip)))
for i in range(len(stores_strip)):
    for k in range(len(skus_strip)):
        out_matrix[i][k] = data[f'{stores_strip[i]}_{skus_strip[k]}']['rmse']


dims = (4.5, 8.27)
plt.rcParams.update({'font.size': .81})
sns.set(font_scale=.71)
fig, ax = plt.subplots(figsize=dims)
ax = sns.heatmap(out_matrix, xticklabels=skus_strip, yticklabels=stores_strip, square=True)
ax.set_xlabel('SKU id')
ax.set_ylabel('Store id')
plt.title(f'ARIMA RMSE for store-sku match, mean: {np.nanmean(out_matrix)}')
plt.tight_layout()
plt.show()


#%%
from imutils import paths
DIR = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp'
pts = list(paths.list_files(DIR))
pts = [p for p in pts if 'whole' in p]

dicts = {}
for p in pts:
    dicts[p.split('/')[-1].split('.')[0]] = pkl.load(open(p, 'rb'))

min_rmse = np.inf
min_out = None
for key in dicts.keys():
    if dicts[key]['rmse_test'] < min_rmse:
        min_rmse = dicts[key]['rmse_test']
        min_out = dicts[key]

print(min_out)
