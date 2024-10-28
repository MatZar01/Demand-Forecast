import json
import math
import os
import pickle
import sys
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import torch
from mpmath.math2 import sqrt2

from capymoa.base import Regressor
from capymoa.stream import NumpyStream, Schema
from capymoa.evaluation import prequential_evaluation_multiple_learners, RegressionEvaluator
from capymoa.evaluation.visualization import plot_windowed_results, plot_predictions_vs_ground_truth
from capymoa.stream.drift import DriftStream, Drift
# from capymoa.drift.detectors import ADWIN

from moa.classifiers.core.driftdetection import ADWIN


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

class DummyDriftStream(DriftStream):
    def __init__(self,
                 # concept_list: list,
                 # max_recurrences_per_concept: int = 3,
                 # transition_type_template: Drift = AbruptDrift(position=5000),
                 # concept_name_list: list = None,
                 # moa_stream = None,
                 # CLI: str = None,
                 X_dummy,
                 y_dummy,
                 drift_indexes = None,
                 concept_info = None,
                 drifts: typing.List[Drift] = None
                 ):
        drifts = []
        if drift_indexes is not None:
            for drift_index in drift_indexes:
                drifts.append(Drift(position=drift_index, width=0))
        self._initial_stream = NumpyStream(np.array(X_dummy).reshape(-1, 1), np.array(y_dummy), target_type='numeric')
        if (
                # concept_info is not None and
                drifts is not None):
            super().__init__(schema=self._initial_stream.get_schema(), drifts=drifts)
            # self.concept_info = concept_info

    def has_more_instances(self):
        return self._initial_stream.has_more_instances()

    def next_instance(self):
        return self._initial_stream.next_instance()

    def get_schema(self):
        return self.schema

    def get_moa_stream(self):
        raise ValueError("Not a moa_stream, FALL stream")

    def restart(self):
        self._initial_stream.restart()



# Function to create tumbling windows
def compute_tumbling_windows(data_frame, window_size):
    # Reshape data into non-overlapping windows
    windows = np.array_split(data_frame, np.arange(window_size, len(data_frame), window_size))
    return windows

# Create the parser
parser = argparse.ArgumentParser()
# Add a string argument
parser.add_argument('-d','--dir', type=str, help="file", default='./')
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

initial_stream = DummyDriftStream(y_hat, true_value)
# cumulative_evaluator = RegressionEvaluator(schema=initial_stream.get_schema())

mse = torch.nn.MSELoss()
# detector = ADWIN(delta=1.0E-2)
detector = ADWIN(1.0E-9)
detection_index = []

for i in range(len(true_value)):
    # cumulative_evaluator.update(true_value[i], y_hat[i])
    # error = cumulative_evaluator.rmse()
    with torch.no_grad():
        error = mse(torch.tensor(y_hat[i]), torch.tensor(true_value[i])).item()
    if i > 0:
        previous_error_estimation = detector.getEstimation()
    # detector.add_element(error)
    detector.setInput(error)
    if detector.getChange():
        if i > 0:
            current_error_estimation = detector.getEstimation()
            if current_error_estimation > previous_error_estimation:
                print(f'Change detected at index: {i}, and error going UP ({current_error_estimation} > {previous_error_estimation}). add index n: {detector.getWidth()}, var: {math.sqrt(detector.getVariance())}')
                detection_index.append(i)
            else:
                print(f'Change detected at index: {i}, and error going DOWN.')
        # cumulative_evaluator = RegressionEvaluator(schema=initial_stream.get_schema())

# detection_index = detector.detection_index
print(f'Total number of detections: {len(detection_index)}')

detection_index = None
initial_stream = DummyDriftStream(y_hat, true_value, drift_indexes=detection_index)



learners = {}
# learners_map = {
#     '( Xn )': ['X', False],
#     '( E(Xc) + Xn )': ['q', False],
#     '( mlp( E(Xc) + Xn) )': ['z', False],
#     '( Xn, y - NN(X) )': ['X', True],
#     '( E(Xc) + Xn,  y - NN(X) )': ['q', True],
#     '( mlp( E(Xc) + Xn),  y - NN(X) )': ['z', True],
# }

for l_name, l in incremental_learners.items():
    ll = DummyClassifier(schema=initial_stream.get_schema(), dataset=l['y_hat'])
    # old_part = str(l_name).replace('SOKNL ', '')
    # new_x = learners_map[old_part][0]
    # residual = learners_map[old_part][1]
    # new_name = ('F_batch(X) + 'if residual else '') + f'SOKNL({new_x})'
    learners[l_name] = ll

learners['F_batch(X)'] = DummyClassifier(schema=initial_stream.get_schema(), dataset=y_hat)

final_results = prequential_evaluation_multiple_learners(initial_stream, learners, store_predictions=True, window_size=WINDOW_SIZE)

for learner_id in learners.keys():
    if learner_id in final_results:
        cumulative = final_results[learner_id]['cumulative']
        final_results[learner_id].learner = f'{learner_id}  cumulative RMSE: {cumulative.rmse():.2f}'

final_results_p = list(final_results.values())
plot_windowed_results(*final_results_p,
                      metric="rmse",
                      plot_aggregated=False, plot_title= f'RMSE (tumbling window size is {WINDOW_SIZE} for periodic)',
                      figure_name='RMSE'
                      )

best_r2 = 0.0
best_inc_learner = None
for learner_id in learners.keys():
    if learner_id in final_results:
        cumulative = final_results[learner_id]['cumulative']
        final_results[learner_id].learner = f'{learner_id} cumulative R2: {cumulative.r2():.2f}'
        if best_r2 < cumulative.r2():
            best_inc_learner = learner_id
            best_r2 = cumulative.r2()
        print(f"{learner_id}, RMSE: {cumulative.rmse():.2f}, R2: {cumulative.r2():.2f}, adjusted R2: {cumulative.adjusted_r2():.2f}")

final_results_p = list(final_results.values())
plot_windowed_results(*final_results_p,
                      metric="r2",
                      plot_aggregated=False, plot_title= f'R2 (tumbling window size is {WINDOW_SIZE} for periodic)',
                      figure_name='R2'
                      )

final_results_g = [v for k, v in final_results.items() if k in ['F_batch(X)', best_inc_learner] ]
plot_predictions_vs_ground_truth(*final_results_g,
                                 ground_truth=np.array(true_value),
                                 plot_interval=(0, len(true_value)-1),
                                 figure_name='PredictionsVsGroundTruth')

plt.show()