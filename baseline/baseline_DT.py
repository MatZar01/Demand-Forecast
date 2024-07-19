import numpy as np
from src import get_args
from src import DataSet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
    data_val = data_manager.val
    # get stores ids and sku ids
    stores = np.unique(data_train[:, 2])
    skus = np.unique(data_train[:, 3])
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
    store_match_train = data_train[np.where(data_train[:, 2] == store_id)[0]]
    train_single = store_match_train[np.where(store_match_train[:, 3] == sku_id)[0]]

    store_match_val = data_val[np.where(data_val[:, 2] == store_id)[0]]
    val_single = store_match_val[np.where(store_match_val[:, 3] == sku_id)[0]]

    # get only relevant data
    train_single_X, train_single_y = train_single[:, [4, 5, 6, 7]], train_single[:, [8]]
    val_single_X, val_single_y = val_single[:, [4, 5, 6, 7]], val_single[:, [8]]
    return train_single_X, train_single_y, val_single_X, val_single_y


def get_lagged_data(X, y, lag, diff):
    X_lagged = []
    y_lagged = []
    y_diff = []
    diff_start = y[lag - 1]
    for i in range(X.shape[0] - (lag-1)-1):
        X_lagged.append(X[i:i+lag, :])
        y_diff.append(y[i + lag] - y[i + lag - 1])
        y_lagged.append(y[i+lag])
    if diff:
        out_y = y_diff
    else:
        out_y = y_lagged
    return np.array(X_lagged).reshape(len(X_lagged), -1), np.array(out_y), diff_start


def get_rmse_score(X, y):
    return sqrt(mean_squared_error(X, y))


def train_forest(X_t, y_t, X_v, y_v, args):
    forest = RandomForestRegressor(n_estimators=args['N_EST'], max_leaf_nodes=args['MAX_LEAF'], n_jobs=-1)
    forest.fit(X_t, y_t)

    preds_train = forest.predict(X_t)
    rmse_train = get_rmse_score(y_t, preds_train)

    preds_val = forest.predict(X_v)
    rmse_val = get_rmse_score(y_v, preds_val)

    return forest, {'rmse_train': rmse_train, 'rmse_val': rmse_val, 'train': [y_t, preds_train], 'val': [y_v, preds_val]}


def train_dt(X_t, y_t, X_v, y_v, args):
    tree = DecisionTreeRegressor(max_depth=args['DEPTH'])
    tree.fit(X_t, y_t)

    preds_train = tree.predict(X_t)
    rmse_train = get_rmse_score(y_t, preds_train)

    preds_val = tree.predict(X_v)
    rmse_val = get_rmse_score(y_v, preds_val)

    return tree, {'rmse_train': rmse_train, 'rmse_val': rmse_val, 'train': [y_t, preds_train], 'val': [y_v, preds_val]}


def prune_dt(X_t, y_t, X_v, y_v, tree):
    alphas = tree.cost_complexity_pruning_path(X_t, y_t).ccp_alphas
    out = {}
    lowest_rmse_val = np.inf
    best_reg = None
    for alpha in alphas:
        tree = DecisionTreeRegressor(ccp_alpha=alpha)
        tree.fit(X_t, y_t)

        preds_train = tree.predict(X_t)
        rmse_train = get_rmse_score(y_t, preds_train)

        preds_val = tree.predict(X_v)
        rmse_val = get_rmse_score(y_v, preds_val)

        if rmse_val < lowest_rmse_val:
            out = {'rmse_train': rmse_train, 'rmse_val': rmse_val, 'train': [y_t, preds_train], 'val': [y_v, preds_val]}
            best_reg = tree
        return best_reg, out


def get_diff_preds(pred_list, diff_start):
    gt = pred_list[0].reshape(pred_list[0].shape[0]).astype(float)
    preds = pred_list[1]
    preds[0] += diff_start
    gt[0] += diff_start
    for i in range(preds.size - 1):
        preds[i + 1] += preds[i]
        gt[i + 1] += gt[i]
    return [gt, preds]


def get_param_range(p_range, d_range, q_range):
    p_s = list(range(0, p_range, 2))
    d_s = list(range(0, d_range))
    q_s = list(range(0, q_range))
    return np.array(np.meshgrid(p_s, d_s, q_s)).T.reshape(-1, 3)


def show_plots(vals, title):
    plt.plot(vals[0], label='GT')
    plt.plot(vals[1], label='preds')
    plt.title(f'{title}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    CFG_FILE = '../cfgs/default.yml'
    SAVE_DIR = './results'
    task_info = get_args(CFG_FILE)
    Data_manager = DataSet(paths=task_info['DATA_PATH'], year_split=True)
    result_dict = {}

    # set parameter grid
    lags = list(range(4, 12, 2))
    depths = list(range(4, 12, 2))
    ests = [100, 150, 200, 250]
    leafs = [10, 15, 20]


    args = {'LAG': 8, 'DEPTH': 9, 'N_EST': 1000, 'MAX_LEAF': 25, 'DIFF': False}

    # get store-sku matches
    c_prod = get_store_sku_match(Data_manager)
    print(f'[INFO] {c_prod.shape[0]} matches found')

    match = c_prod[0]
    train_single_X, train_single_y, val_single_X, val_single_y = get_data(Data_manager, match)
    train_x_lag, train_y_lag, diff_train_start = get_lagged_data(train_single_X, train_single_y, args['LAG'], args['DIFF'])
    val_x_lag, val_y_lag, diff_val_start = get_lagged_data(val_single_X, val_single_y, args['LAG'], args['DIFF'])

    # train DT
    diffs = [diff_train_start, diff_val_start]
    tree, out = train_dt(train_x_lag, train_y_lag, val_x_lag, val_y_lag, args=args)

    # prune DT
    tree_pruned, out_pruned = prune_dt(train_x_lag, train_y_lag, val_x_lag, val_y_lag, tree)

    # random forests
    forest, out_forest = train_forest(train_x_lag, train_y_lag, val_x_lag, val_y_lag, args)

    if args['DIFF']:
        # data no diff
        tx, ty, td = get_lagged_data(train_single_X, train_single_y, args['LAG'], args['DIFF'])
        vx, xy, vd = get_lagged_data(val_single_X, val_single_y, args['LAG'], args['DIFF'])
        initial_preds = get_diff_preds(out['val'], vd)
        pruned_preds = get_diff_preds(out_pruned['val'], vd)
        forest_preds = get_diff_preds(out_forest['val'], vd)

    else:
        initial_preds = out['val']
        pruned_preds = out_pruned['val']
        forest_preds = out_forest['val']

    show_plots(initial_preds, 'Initial')
    show_plots(pruned_preds, 'Pruned')
    show_plots(forest_preds, 'Random Forest')

    print(f'Initial tree RMSE:\nTrain: {out["rmse_train"]}\nVal: {out["rmse_val"]}\n'
          f'Pruned tree RMSE:\nTrain: {out_pruned["rmse_train"]}\nVal: {out_pruned["rmse_val"]}')

    print(f'Random Forest RMSE:\nTrain: {out_forest["rmse_train"]}\nVal: {out_forest["rmse_val"]}')

