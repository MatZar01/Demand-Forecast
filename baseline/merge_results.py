

import os
import pickle

args_dir = '/Users/nuwan.gunasekara/Desktop/CODE/Demand-Forecast/DS/demand-forecasting_LAG_0'
with open(os.path.join(args_dir, 'results_half.pkl'), 'rb') as f:
    results1 = pickle.load(f)

args_dir = '/Users/nuwan.gunasekara/Desktop/CODE/Demand-Forecast/DS/demand-forecasting_LAG_0_1'
with open(os.path.join(args_dir, 'results.pkl'), 'rb') as f:
    results2 = pickle.load(f)


for l_name, l in results2['incremental_learners'].items():
    results1['incremental_learners'][l_name] = l


args_dir = '/Users/nuwan.gunasekara/Desktop/CODE/Demand-Forecast/DS/demand-forecasting_LAG_0'
with open(os.path.join(args_dir, 'results.pkl'), 'wb') as f:
    pickle.dump(results1, f)
