import numpy as np
import pickle

"""
CLUSTERING 25/50/100
"""

data_cluster = pickle.load(open('/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/model_out_clustering.pkl', 'rb'))
data_nums = pickle.load(open('/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/emb_assignments.pkl', 'rb'))

# 25 models
rmses = data_cluster[25]
nums = data_nums[25]
rmse_25_w = 0
rmse_25_nums = 0
for i in range(len(rmses)):
      rmse_25_w += rmses[i] * len(nums[i])
      rmse_25_nums += len(nums[i])

mean_rmse_25 = rmse_25_w/rmse_25_nums

# 50 models
rmses = data_cluster[50]
nums = data_nums[50]
rmse_50_w = 0
rmse_50_nums = 0
for i in range(len(rmses)):
      rmse_50_w += rmses[i] * len(nums[i])
      rmse_50_nums += len(nums[i])

mean_rmse_50 = rmse_50_w/rmse_50_nums

# 100 models
rmses = data_cluster[100]
nums = data_nums[100]
rmse_100_w = 0
rmse_100_nums = 0
for i in range(len(rmses)):
      rmse_100_w += rmses[i] * len(nums[i])
      rmse_100_nums += len(nums[i])

mean_rmse_100 = rmse_100_w/rmse_100_nums

print(f'CLUSTERING RESULTS 25-50-100\n'
      f'25 models: {mean_rmse_25}\n'
      f'50 models: {mean_rmse_50}\n'
      f'100 models: {mean_rmse_100}\n\n')

#%%
import numpy as np
import pickle

"""
CLUSTERING 10/15/20
"""

data_cluster = pickle.load(open('/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/model_out_clustering_10_15_20.pkl', 'rb'))
data_nums = pickle.load(open('/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/emb_assignments_10_15_20.pkl', 'rb'))

# 10 models
rmses = data_cluster[10]
nums = data_nums[10]
rmse_10_w = 0
rmse_10_nums = 0
for i in range(len(rmses)):
      rmse_10_w += rmses[i] * len(nums[i])
      rmse_10_nums += len(nums[i])

mean_rmse_10 = rmse_10_w/rmse_10_nums

# 15 models
rmses = data_cluster[15]
nums = data_nums[15]
rmse_15_w = 0
rmse_15_nums = 0
for i in range(len(rmses)):
      rmse_15_w += rmses[i] * len(nums[i])
      rmse_15_nums += len(nums[i])

mean_rmse_15 = rmse_15_w/rmse_15_nums

# 25 models
rmses = data_cluster[20]
nums = data_nums[20]
rmse_20_w = 0
rmse_20_nums = 0
for i in range(len(rmses)):
      rmse_20_w += rmses[i] * len(nums[i])
      rmse_20_nums += len(nums[i])

mean_rmse_20 = rmse_20_w/rmse_20_nums

print(f'CLUSTERING RESULTS 10-15-20\n'
      f'10 models: {mean_rmse_10}\n'
      f'15 models: {mean_rmse_15}\n'
      f'20 models: {mean_rmse_20}\n\n')
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

#%%
import pickle
import numpy as np


"""
MODEL REFINEMENT
"""
data_refined = pickle.load(open('/home/mateusz/Desktop/Demand-Forecast/baseline/results/name_clustering/model_grouping_refinement.pkl', 'rb'))

rmse_w = 0
weights = 0

for key in data_refined.keys():
      rmse_w += data_refined[key]['new_rmse'] * len(data_refined[key]['matches'])
      weights += len(data_refined[key]['matches'])

print(f'GROUPING REFINEMENT RESULTS\n'
      f'NUMBER OF MODELS: {len(data_refined.keys())}\n'
      f'Mean rmse: {rmse_w/weights}')
