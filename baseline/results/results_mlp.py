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
path_mlp = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/L_15_Q_True_EM_False.pkl'
data_mlp = get_data(path_mlp)
out_mlp = get_out_matrix(data_mlp)
show_matrix(out_mlp, data_mlp)
#%%
path_mlp = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/L_15_Q_True_EM_True.pkl'
data_mlp = get_data(path_mlp)
out_mlp = get_out_matrix(data_mlp)
show_matrix(out_mlp, data_mlp)