import numpy as np
from src import get_args
from src import DataSet, DataSet_3
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime
import pickle as pkl
import warnings
from tqdm import tqdm


def get_timestamp():
    now = datetime.now()
    return f'{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}:{now.second}'


def get_store_sku_match(data_manager):
    data_train = data_manager.train_all
    # get stores ids and sku ids
    stores = np.unique(data_train[:, 1])
    skus = np.unique(data_train[:, 2])
    # get cartesian product for store-sku
    c_prod = np.transpose([np.tile(stores, len(skus)), np.repeat(skus, len(stores))])
    # pick one pair for experiments
    return c_prod


def get_data(data_manager, ids: list):
    store_id = ids[0]
    sku_id = ids[1]
    data_train = data_manager.train_all
    data_val = data_manager.val
    # get all data from selected pair for training and validation
    store_match_train = data_train[np.where(data_train[:, 1] == store_id)[0]]
    train_single = store_match_train[np.where(store_match_train[:, 2] == sku_id)[0]]

    store_match_val = data_val[np.where(data_val[:, 1] == store_id)[0]]
    val_single = store_match_val[np.where(store_match_val[:, 2] == sku_id)[0]]

    # get only relevant data
    train_single = train_single[:, [-2, -1]]
    val_single = val_single[:, [-2, -1]]
    return train_single, val_single


def get_data_series(train_single, val_single):
    train_series = pd.Series(train_single[:, 1].astype(float))
    train_series.index = pd.Index(pd.to_datetime(train_single[:, 0], dayfirst=True, format="mixed"))
    # train_series.index = train_series.index.to_period('M')
    val_series = pd.Series(val_single[:, 1].astype(float))
    val_series.index = pd.Index(pd.to_datetime(val_single[:, 0], dayfirst=True, format="mixed"))
    return train_series, val_series


def get_param_range(p_range, d_range, q_range):
    p_s = list(range(0, p_range, 2))
    d_s = list(range(0, d_range))
    q_s = list(range(0, q_range))
    return np.array(np.meshgrid(p_s, d_s, q_s)).T.reshape(-1, 3)


def train_model(train_series, val_series, order, p_bar):
    if train_series.size > 1500:
        train_series = train_series[-1500:]
    if val_series.size > 1000:
        val_series = val_series[:1000]
    history = [x for x in train_series]

    predictions = []
    # walk-forward validation
    try:  # for incompatible parameters
        for t in range(len(val_series)):
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            output = model_fit.forecast()
            pred = output[0]
            predictions.append(pred)
            history.append(val_series.values[t])
    except:
        print('[INFO -- WARN] training failed')

    try:
        rmse = sqrt(mean_squared_error(val_series.values[:len(predictions)], predictions))  # [:len(predictions)] for fail during forecasting
        p_bar.update(1)
        return rmse, predictions, val_series.values[:len(predictions)]

    except:
        print('[INFO] RMSE failed')
        p_bar.update(1)
        return np.nan, None, None


def grid_search_models(train_series, val_series, param_grid, p_bar):
    best_rmse = np.inf
    best_results = [None, None, None]
    best_params = None
    for order in param_grid:
        model_results = train_model(train_series, val_series, order, p_bar)
        if model_results[0] < best_rmse:
            best_rmse = model_results[0]
            best_results = model_results
            best_params = {'p': order[0], 'd': order[1], 'q': order[2]}

    if best_rmse == np.inf:
        print('[INFO -- WARN] no convergence')
        best_results = [np.nan, None, val_series.values]

    print(f'[INFO] training complete, RMSE: {best_rmse}')
    return {'model': best_params, 'rmse': best_results[0], 'preds': best_results[1], 'gt': best_results[2]}


def save_results(save_dir, result_dict):
    ts = get_timestamp()
    pkl.dump(result_dict, open(f'{save_dir}/ARIMA_3_{ts}.pkl', 'wb'))
    print('[INFO] results saved')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    CFG_FILE = '../cfgs/default.yml'
    SAVE_DIR = './results'
    task_info = get_args(CFG_FILE)
    Data_manager = DataSet_3(paths=task_info['DATA_PATH'], year_split=True)

    # get store-sku matches
    c_prod = get_store_sku_match(Data_manager)
    print(f'[INFO] {c_prod.shape[0]} matches found')

    # select range of model parameters
    p_range = 3
    d_range = 1
    q_range = 1
    param_grid = get_param_range(p_range, d_range, q_range)

    # setup progress bar
    p_bar = tqdm(range(c_prod.shape[0]*param_grid.shape[0]))  # total 57456 models

    # perform full training
    result_dict = {}
    for match in c_prod:
        train_single, val_single = get_data(Data_manager, match)
        train_series, val_series = get_data_series(train_single, val_single)
        best_model_res = grid_search_models(train_series, val_series, param_grid, p_bar)
        result_dict[f'{match[0]}_{match[1]}'] = best_model_res

    # save results
    save_results(SAVE_DIR, result_dict)
    print('[INFO] DONE')
