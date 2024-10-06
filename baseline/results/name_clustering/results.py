import numpy as np
import pickle

"""
CLUSTERING
"""

data_cluster = pickle.load(open('/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/model_out_clustering.pkl', 'rb'))

# 25 models
rmses = data_cluster[25]
mean_rmse_25 = np.mean(rmses)
# 50 models
rmses = data_cluster[50]
mean_rmse_50 = np.mean(rmses)
# 100 models
rmses = data_cluster[100]
mean_rmse_100 = np.mean(rmses)

print(f'CLUSTERING RESULTS\n'
      f'25 models: {mean_rmse_25}\n'
      f'50 models: {mean_rmse_50}\n'
      f'100 models: {mean_rmse_100}\n\n')

#%%
import pickle
import numpy as np


"""
MODEL GROUPING
"""
data_grouping = pickle.load(open('/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/model_out_grouper_1.pkl', 'rb'))

rmse_w = 0
weights = 0

for key in data_grouping.keys():
      rmse_w += data_grouping[key]['rmse'] * len(data_grouping[key]['matches'])
      weights += len(data_grouping[key]['matches'])

print(f'GROUPING RESULTS\n'
      f'NUMBER OF MODELS: {len(data_grouping.keys())}\n'
      f'Mean rmse: {rmse_w/weights}')
