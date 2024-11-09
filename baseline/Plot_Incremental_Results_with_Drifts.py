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


# Create the parser
parser = argparse.ArgumentParser()
# Add a string argument
parser.add_argument('-d','--dir', type=str, help="file", default='./')
parser.add_argument('-D', '--detect_drift', action='store_true', help="Detect Drift")
# Parse the arguments
args = parser.parse_args()

with open(os.path.join(args.dir, 'all_seed_results.pkl'), 'rb') as f:
    print(f'Reading results from {f.name}')
    results = pickle.load(f)

with open(os.path.join(args.dir, 'config.json') , 'r') as file:
    config = json.load(file)

DETECT_DRIFTS = args.detect_drift

WINDOW_SIZE = config['WINDOW_SIZE']
ADWIN_delta = config['ADWIN_delta']

true_value = results['true_value']
y_hat = results['y_hat']

initial_stream = DummyDriftStream(y_hat, true_value)

detection_index = None
if DETECT_DRIFTS:
    mse = torch.nn.MSELoss()
    detector = ADWIN(ADWIN_delta)
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

initial_stream = DummyDriftStream(y_hat, true_value, drift_indexes=detection_index)

learners = {}
inc_learner = ['X', 'q', 'z']

for idx, incremental_learners in enumerate(results['incremental_learners_per_seed']):
    for l_name, l in incremental_learners.items():
        ll = DummyClassifier(schema=initial_stream.get_schema(), dataset=l['y_hat'])
        k = f'F_inc({l_name}):{idx}'
        k = str(k).replace('F_inc(F_batch', 'F_batch').replace('))', ')')
        learners[k] = ll

learners['F_batch(X)'] = DummyClassifier(schema=initial_stream.get_schema(), dataset=y_hat)

final_results = prequential_evaluation_multiple_learners(initial_stream, learners, store_predictions=True, window_size=WINDOW_SIZE, max_instances=len(true_value))
final_results_p = []
for learner_id in learners.keys():
    learner_id_str = str(learner_id).split(':')[0]
    final_results[learner_id].learner = learner_id_str
    final_results_p.append(final_results[learner_id])

# final_results_p = list(final_results.values())

cumulative_df = plot_windowed_results(*final_results_p,
                      metric="rmse",
                      ylabel='RMSE',
                      plot_aggregated=True,
                      plot_title= f'RMSE (tumbling window size is {WINDOW_SIZE} for periodic).' + (f'ADWIN delta: {ADWIN_delta:.2e}' if DETECT_DRIFTS else ''),
                      figure_name='RMSE' + ('_DD'if DETECT_DRIFTS else '')
                      )
print(cumulative_df.to_latex(index=False, float_format="%.2f"))

cumulative_df = plot_windowed_results(*final_results_p,
                      metric="r2",
                      ylabel='R2',
                      plot_aggregated=True,
                      plot_title= f'R2 (tumbling window size is {WINDOW_SIZE} for periodic).' + (f'ADWIN delta: {ADWIN_delta:.2e}' if DETECT_DRIFTS else ''),
                      figure_name='R2' + ('_DD'if DETECT_DRIFTS else '')
                      )
print(cumulative_df.to_latex(index=False, float_format="%.2f"))

best_inc_learner = cumulative_df.loc[cumulative_df['mean'].idxmax(), 'learner']
best_inc_learner_array = []

for idx, incremental_learners in enumerate(results['incremental_learners_per_seed']):
    for l_name, l in incremental_learners.items():
        if 'F_batch(X) + ' in best_inc_learner:
            if l_name == best_inc_learner:
                best_inc_learner_array.append(l['y_hat'])
        else:
            if l_name == str(best_inc_learner).replace('F_inc', '').replace('(', '').replace(')', ''):
                best_inc_learner_array.append(l['y_hat'])


best_learner = {}
initial_stream = DummyDriftStream(y_hat, true_value, drift_indexes=detection_index)
best_learner[f"{best_inc_learner} R2: {round(cumulative_df.loc[cumulative_df['mean'].idxmax(), 'mean'], 2)} "] = DummyClassifier(schema=initial_stream.get_schema(), dataset=list(np.array(best_inc_learner_array).mean(axis=0)))
best_learner['F_batch(X)'] = DummyClassifier(schema=initial_stream.get_schema(), dataset=y_hat)
best_results = prequential_evaluation_multiple_learners(initial_stream, best_learner, store_predictions=True, window_size=WINDOW_SIZE, max_instances=len(true_value))

for learner_id in best_results.keys():
    best_results[learner_id].learner = learner_id

best_results = list(best_results.values())
plot_predictions_vs_ground_truth(*best_results,
                                 ground_truth=np.array(true_value),
                                 plot_interval=(0, len(true_value)-1),
                                 figure_name='PredictionsVsGroundTruth')

plt.show()