import os
import pickle

# args_dir = '/Users/ng98/Desktop/CODE/Demand-Forecast/demand-forecasting'

with open(os.path.join(args_dir, 'results.pkl'), 'rb') as f:
    results1 = pickle.load(f)

# args_dir = '/Users/ng98/Desktop/CODE/Demand-Forecast/demand-forecasting_2'
with open(os.path.join(args_dir, 'results.pkl'), 'rb') as f:
    results2 = pickle.load(f)

# del results1['incremental_learners']['SOKNL ( Xn, y - NN(X) )']

for k in results2['incremental_learners'].keys():
    results1['incremental_learners'][k] = results2['incremental_learners'][k]

# args_dir = '/Users/ng98/Desktop/CODE/Demand-Forecast/DS/demand-forecasting_final'
with open(os.path.join(args_dir, 'results.pkl'), 'wb') as f:
    pickle.dump(results1, f)