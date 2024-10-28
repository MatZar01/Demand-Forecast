import os
import pickle

from joblib.externals.cloudpickle import instance

# from cProfile import label
# from copy import deepcopy

# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error

from base_src import MLP_dataset_emb
from base_src import MLP, MLP_emb, MLP_emb_tl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from capymoa.drift.detectors import ADWIN, DDM
from capymoa.stream import NumpyStream
from capymoa.regressor import SGBR,SOKNL, AdaptiveRandomForestRegressor
from capymoa.evaluation import RegressionEvaluator


import math

import json
import argparse

# Create the parser
parser = argparse.ArgumentParser()
# Add a string argument
parser.add_argument('-d','--dir', type=str, help="file", default='../DS/NZ_energy_latest_LAG_15_residual')
# Parse the arguments
args = parser.parse_args()



class NormalizationStats:
    def __init__(self):
        self.examples_seen = 0
        self.sum_of_values = 0.0
        self.sum_of_squares = 0.0

    def update(self, value):
        self.examples_seen += 1
        self.sum_of_values += value
        self.sum_of_squares += value * value

    def reset(self):
        self.examples_seen = 0
        self.sum_of_values = 0.0
        self.sum_of_squares = 0.0


def normalize_target_value(value, examples_seen, sum_of_values, sum_of_squares):
# if examples_seen > 1:
    standard_deviation = math.sqrt((sum_of_squares - ((sum_of_values * sum_of_values) / examples_seen)) / examples_seen)
    average = sum_of_values / examples_seen
    if standard_deviation > 0:
        return (value - average) / (3 * standard_deviation)
    return 0.0
# else:
    #     return 0.0

def get_normalized_error(true_value, prediction, stats: NormalizationStats):
    examples_seen = stats.examples_seen
    sum_of_values = stats.sum_of_values
    sum_of_squares = stats.sum_of_squares

    if examples_seen > 1:
        try:
            normalized_value = normalize_target_value(true_value, examples_seen, sum_of_values, sum_of_squares)
            normalized_prediction = normalize_target_value(prediction, examples_seen, sum_of_values, sum_of_squares)
            return abs(normalized_value - normalized_prediction)
        except Exception as e:
            print(f'Error: {e}')
            return 0.0
    return 0.0

# class RMSELoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = torch.nn.MSELoss()
#
#     def forward(self, yhat, y):
#         return torch.sqrt(self.mse(yhat, y))


"""
Hi, in this simple file I'll walk you through the inference with the models
models should be in ../inference_tests_models as .pth files 
If you'd have questions, feel free to ask me on discord
"""
# Reading config from JSON file
config = None
with open(os.path.join(args.dir, 'config.json') , 'r') as file:
    config = json.load(file)

# DEVICE = 'cuda'  # can be set to 'cpu', in this script it won't matter anyhow
DEVICE = 'cpu'  # can be set to 'cpu', in this script it won't matter anyhow
BATCH = 1  # batch is set to 1 just for convenience
LAG = 15  # how many samples are in the series -- it depends on the model architecture, so 15 it is
# LAG = 1  # how many samples are in the series -- it depends on the model architecture, so 15 it is
# QUANT = True  # if to add previous sales to input vector
QUANT = False  # if to add previous sales to input vector. This ia already available in NZ Energy
EMBED = True  # if to use embedding (onehot embedding to int for cat2vec embedder)
NORMALIZE = True  # if to normalize the rest of input vector
MATCHES_ONLY = False  # if to only select single SKU-Store match from dataloader - affects m = None

WINDOW_SIZE = 500 # for plotting
RANDOM_TRAIN_INC = True
INC_TRAIN_WITH_EMBEDDINGS = True

if not MATCHES_ONLY:
    m = None

# select train .csv from dataset (in test .csv there are no labels)
# DATA_PATH = '../DS/demand-forecasting/train.csv'
DATA_PATH = f"{os.path.join(args.dir, config['data'])}"

# embedders are only used for onehot-to-int encoding for cat2vec -- they are saved to omit confusion between models
# embedders = {'C2': {'onehot': '../embedding_models/onehot_C2.pkl'},
#              'C3': {'onehot': '../embedding_models/onehot_C3.pkl'}}

embedders = {'C2': {'onehot': f"{os.path.join(args.dir, config['C2'])}"},
             'C3': {'onehot': f"{os.path.join(args.dir, config['C3'])}"}}


inc_types = [
    # 'X',
    # 'q',
    # 'z',
    'F_batch(X) + F_inc(X)',
    'F_batch(X) + F_inc(q)',
    'F_batch(X) + F_inc(z)'
]

inc_learners = {
    # 'SGBR': [SGBR, {'base_learner': 'meta.OzaBag -s 10 -l (trees.FIMTDD -s VarianceReductionSplitCriterion -g 50 -c 0.01 -e)', 'boosting_iterations': 10}],
    'SOKNL': [SOKNL, {}],
    # 'ARFR': [AdaptiveRandomForestRegressor, {}]
}
incremental_learners = {}

for k,v in inc_learners.items():
    for inc_type in inc_types:
        incremental_learners[f'{k} ( {inc_type} )'] = {
            'model': None,
            'model_f': v[0],
            'model_f_para': v[1],
            'last_prediction': 0.0,
            'y_hat': [],
            'y_hat_none_at': [],
            'y_hat_nan_at': [],
            'instance_none_at': [],
            'evaluator': None,
            'feature_vec_none_at': [],
            # 'init_stream': None,
            'type': inc_type
        }


true_value = []
y_hat = []
y_hat_none_at = []
y_hat_nan_at = []
idxs = []
last_prediction = 0.0
evaluator = None

def train_incremental_and_predict(model, dataloader, evaluator, predict=False):
    i = 0
    # Iterating over the data loader
    for emb_col_2, emb_col_3, feature_vec, y in tqdm(dataloader):
        i += 1

        with torch.no_grad():
            # get output from model
            feature_vec_np = torch.flatten(feature_vec, start_dim=1).detach().numpy()
            concat_embeddings = model.get_concat_embeddings(emb_col_2, emb_col_3, feature_vec)
            concat_embeddings_np = concat_embeddings.detach().numpy()
            logits = model.get_logits(concat_embeddings)
            logits_np = logits.detach().numpy()
            y_hat_tensor = model.clf(logits)

            y_hat_float = y_hat_tensor.detach().numpy()[0].item()
            y_numpy = torch.flatten(y).detach().numpy()
            y_float = y_numpy.item()


        for l, inc_model in incremental_learners.items():
            stream = None
            if feature_vec is not None: # sometimes we get None feature vectors, check against that
                # Train incremental models
                initial_prediction = 0.0
                if inc_model['type'] == 'X':
                    stream = NumpyStream(feature_vec_np, y_numpy, target_type='numeric')
                elif inc_model['type'] == 'q':
                    stream = NumpyStream(concat_embeddings_np, y_numpy, target_type='numeric')
                elif inc_model['type'] == 'z':
                    stream = NumpyStream(logits_np, y_numpy, target_type='numeric')
                elif inc_model['type'] == 'F_batch(X) + F_inc(X)':
                    initial_prediction = y_hat_float
                    stream = NumpyStream(feature_vec_np, y_numpy-y_hat_float, target_type='numeric')
                elif inc_model['type'] == 'F_batch(X) + F_inc(q)':
                    initial_prediction = y_hat_float
                    stream = NumpyStream(concat_embeddings_np, y_numpy-y_hat_float, target_type='numeric')
                elif inc_model['type'] == 'F_batch(X) + F_inc(z)':
                    initial_prediction = y_hat_float
                    stream = NumpyStream(logits_np, y_numpy-y_hat_float, target_type='numeric')

                # if inc_model['init_stream'] is None:
                #     inc_model['init_stream'] = stream
                if inc_model['model'] is None:
                    para = inc_model['model_f_para']
                    para['schema'] = stream.get_schema()
                    inc_model['model'] = inc_model['model_f'](**para)

            # test then train incremental model
                instance = stream.next_instance()
                if instance is not None:
                    try:
                        if predict:
                            prediction_inc = initial_prediction + inc_model['model'].predict(instance)
                            prediction_inc = float(prediction_inc)
                            if math.isnan(prediction_inc):
                                prediction_inc = inc_model['last_prediction']
                                inc_model['y_hat_nan_at'].append(i)
                            inc_model['last_prediction'] = prediction_inc
                            inc_model['y_hat'].append(inc_model['last_prediction'])
                        inc_model['model'].train(instance)
                    except:
                        if predict:
                            inc_model['y_hat_none_at'].append(i)
                            inc_model['y_hat'].append(inc_model['last_prediction'])
                else:
                    if predict:
                        inc_model['instance_none_at'].append(i)
                        inc_model['y_hat'].append(inc_model['last_prediction'])
            else:  # feature vector is None
                if predict:
                    for l, inc_model in incremental_learners.items():
                        inc_model['feature_vec_none_at'].append(i)
                        inc_model['y_hat'].append(inc_model['last_prediction'])
            if predict:
                # Update evaluators
                if inc_model['evaluator'] is None:
                    inc_model['evaluator'] = RegressionEvaluator(schema=stream.get_schema())
                inc_model['evaluator'].update(y_float, inc_model['last_prediction'])

        if predict:
            if y_hat_float is None:
                y_hat_none_at.append(i)
            if math.isnan(y_hat_float):
                y_hat_float = last_prediction
                y_hat_nan_at.append(i)
            true_value.append(y_float)
            y_hat.append(y_hat_float)

            # Update evaluator
            if evaluator is None:
                initial_stream = NumpyStream(feature_vec_np, y_numpy, target_type='numeric')
                evaluator = RegressionEvaluator(schema=initial_stream.get_schema())
            evaluator.update(y_float, 0.0 if y_hat_float is None or math.isnan(y_hat_float) else y_hat_float)

            idxs.append(i)

    return evaluator


if __name__ == '__main__':
    # load the model - there are two: mlp_model_for_ft was for finetuning, mlp_model_for_tl was for transfer learning
    # they both work the same, only difference is that mlp_model_for_tl has exchangeable classifier layer
    # model = torch.load('../inference_tests_models/mlp_model_for_tl.pth')
    pre_trained_model = torch.load(f"{os.path.join(args.dir, config['model'])}")

    # test then train incremental learner with train data
    # train option does not have an effect
    train_data = MLP_dataset_emb(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                               embedders=embedders, matches=m, data_split='train')
    train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=RANDOM_TRAIN_INC, num_workers=0)
    _ = train_incremental_and_predict(pre_trained_model, train_dataloader,None, predict=False)

    # test then train incremental learner with val data
    val_data = MLP_dataset_emb(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                               embedders=embedders, matches=m, data_split='val')
    val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=RANDOM_TRAIN_INC, num_workers=0)
    _ = train_incremental_and_predict(pre_trained_model, val_dataloader, None, predict=False)

    # test then train incremental learner with test data
    test_data = MLP_dataset_emb(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                               embedders=embedders, matches=m, data_split='test')
    test_dataloader = DataLoader(test_data, batch_size=BATCH, shuffle=False, num_workers=0)
    true_value = []
    y_hat = []
    y_hat_none_at = []
    y_hat_nan_at = []
    idxs = []
    last_prediction = 0.0
    g_evaluator = None
    g_evaluator = train_incremental_and_predict(pre_trained_model, test_dataloader, g_evaluator, predict=True)


    for l, inc_model in incremental_learners.items():
        inc_model['evaluator_results'] = {
            'adjusted_r2': round(float(inc_model['evaluator'].adjusted_r2()), 2),
            'r2': round(float(inc_model['evaluator'].r2()), 2),
            'rmse': round(float(inc_model['evaluator'].rmse()), 2)}
        inc_model['evaluator'] = None
        inc_model['model'] = None
        inc_model['model_f'] = None
        inc_model['model_f_para'] = None
        # inc_model['y_hat_nan_at'] = np.where(np.isnan(np.array(inc_model['y_hat'])))[0],


    results = {
        'config': config,
        'WINDOW_SIZE': WINDOW_SIZE,
        'incremental_learners': incremental_learners,
        'true_value': true_value,
        'y_hat': y_hat,
        'y_hat_none_at': y_hat_none_at,
        # 'y_hat_nan_at': np.where(np.isnan(np.array(y_hat)))[0],
        'y_hat_nan_at': y_hat_nan_at,
        'idxs': idxs,
        'evaluator_results': {
            'adjusted_r2': round(float(g_evaluator.adjusted_r2()), 2),
            'r2': round(float(g_evaluator.r2()), 2),
            'rmse': round(float(g_evaluator.rmse()), 2)},
        'RANDOM_TRAIN_INC': RANDOM_TRAIN_INC
    }

    with open(os.path.join(args.dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
