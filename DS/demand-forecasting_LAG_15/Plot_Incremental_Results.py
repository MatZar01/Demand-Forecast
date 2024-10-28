import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from capymoa.base import Regressor
from capymoa.stream import NumpyStream
from capymoa.evaluation import prequential_evaluation_multiple_learners
from capymoa.evaluation.visualization import plot_windowed_results, plot_predictions_vs_ground_truth


class DummyClassifier(Regressor):
    def __init__(self, schema=None, random_seed=1, dataset=None):
        self.dataset = dataset
        self.index = 0
        super().__init__(schema, random_seed)

    def train(self, instance):
        pass

    def predict(self, instance):
        pred = self.dataset[self.index]
        self.index += 1
        return pred

    def __str__(self):
        return str('DummyClassifier')



# Function to create tumbling windows
def compute_tumbling_windows(data_frame, window_size):
    # Reshape data into non-overlapping windows
    windows = np.array_split(data_frame, np.arange(window_size, len(data_frame), window_size))
    return windows

# Create the parser
parser = argparse.ArgumentParser()
# Add a string argument
parser.add_argument('-d','--dir', type=str, help="file", default='/Users/ng98/Desktop/CODE/Demand-Forecast/DS/demand-forecasting_LAG_15')
# Parse the arguments
args = parser.parse_args()


with open(os.path.join(args.dir, 'config.json') , 'r') as file:
    config = json.load(file)


with open(os.path.join(args.dir, 'results.pkl'), 'rb') as f:
    results = pickle.load(f)


WINDOW_SIZE = results['WINDOW_SIZE']
WINDOW_SIZE = 1000
# drift_detectors = results['drift_detectors']
incremental_learners = results['incremental_learners']
true_value = results['true_value']
y_hat = results['y_hat']
y_hat_none_at = results['y_hat_none_at']
# 'y_hat_nan_at': np.where(np.isnan(np.array(y_hat)))[0],
y_hat_nan_at = results['y_hat_nan_at']
idxs = results['idxs']
RANDOM_TRAIN_INC = results['RANDOM_TRAIN_INC']
evaluator_results = results['evaluator_results']




initial_stream = NumpyStream(np.array(y_hat).reshape(-1,1), np.array(true_value), target_type='numeric')

learners = {}

for l_name, l in incremental_learners.items():
    ll = DummyClassifier(schema=initial_stream.get_schema(), dataset=l['y_hat'])
    learners[l_name] = ll

learners['MLP'] = DummyClassifier(schema=initial_stream.get_schema(), dataset=y_hat)

final_results = prequential_evaluation_multiple_learners(initial_stream, learners, store_predictions=True)

for learner_id in learners.keys():
    if learner_id in final_results:
        final_results[learner_id].learner = learner_id
        cumulative = final_results[learner_id]['cumulative']
        print(f"{learner_id}, RMSE: {cumulative.rmse():.2f}, R2: {cumulative.r2():.2f}, adjusted R2: {cumulative.adjusted_r2():.2f}")

final_results_p = list(final_results.values())
plot_windowed_results(*final_results_p,  metric="rmse")
plot_windowed_results(*final_results_p,  metric="r2")

final_results_g = [v for k, v in final_results.items() if k in ['MLP', 'SOKNL ( E(Xc) + Xn,  y - NN(X) )'] ]
plot_predictions_vs_ground_truth(*final_results_g, ground_truth=np.array(true_value), plot_interval=(0, len(true_value)-1))

plt.show()