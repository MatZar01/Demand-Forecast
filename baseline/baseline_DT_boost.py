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
import xgboost as xgb


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


def show_plots(vals, title):
    plt.plot(vals[0], label='GT')
    plt.plot(vals[1], label='preds')
    plt.title(f'{title}')
    plt.legend()
    plt.show()


def save_result(res_dict, name, save_dir):
    ts = get_timestamp()
    pkl.dump(res_dict, open(f'{save_dir}/{name}_{ts}.pkl', 'wb'))


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
    grid = np.array(np.meshgrid(lags, depths)).T.reshape(-1, 2)

    args = {'LAG': 8, 'DEPTH': 4, 'N_EST': 200, 'MAX_LEAF': 15, 'DIFF': False}

    # get store-sku matches
    c_prod = get_store_sku_match(Data_manager)
    print(f'[INFO] {c_prod.shape[0]} matches found')

    match = c_prod[0]
    train_single_X, train_single_y, val_single_X, val_single_y = get_data(Data_manager, match)
    train_x_lag, train_y_lag, diff_train_start = get_lagged_data(train_single_X, train_single_y, args['LAG'], args['DIFF'])
    val_x_lag, val_y_lag, diff_val_start = get_lagged_data(val_single_X, val_single_y, args['LAG'], args['DIFF'])

    # train DT
    diffs = [diff_train_start, diff_val_start]
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:squarederror',
                           max_depth=9,
                           learning_rate=0.01)

    reg.fit(train_x_lag, train_y_lag, eval_set=[(train_x_lag, train_y_lag), (val_x_lag, val_y_lag)], verbose=100)
    show_plots([val_y_lag, reg.predict(val_x_lag)], 'Boosted')
