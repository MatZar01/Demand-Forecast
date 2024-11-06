

import os
import pickle

base_dir = '/Users/ng98/Desktop/CODE/Demand-Forecast/DS/demand-forecasting_LAG_0'
seeds = [1, 2, 3, 4, 5]

results = None
for r in seeds:
    d = f'{base_dir}_{r}'
    with open(os.path.join(d, 'results.pkl'), 'rb') as f:
        temp_results = pickle.load(f)
        if results is None:
            results = temp_results
            results['incremental_learners_per_seed'] = []
        results['incremental_learners_per_seed'].append(temp_results['incremental_learners'])


with open(os.path.join(f'{base_dir}_{1}', 'all_seed_results.pkl'), 'wb') as f:
    pickle.dump(results, f)
