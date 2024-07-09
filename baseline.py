import numpy as np
import torch
from src import get_args

CFG_FILE = 'cfgs/default.yml'
task_info = get_args(CFG_FILE)
#%%
from src import DataSet
Data_manager = DataSet(paths=task_info['DATA_PATH'])
data_train = Data_manager.train[:, 1:]
data_val = Data_manager.val[:, 1:]
#%%
weeks = Data_manager.data_merged[:, 1]
store_ids = Data_manager.data_merged[:, 2]
sku_ids = Data_manager.data_merged[:, 3]
unq_weeks = np.unique(weeks, return_counts=True)
unq_store_ids = np.unique(store_ids, return_counts=True)
unq_sku_ids = np.unique(sku_ids, return_counts=True)
#%%
d_unsh = Data_manager.data_unshuffled
week_data = {'11': np.zeros((13, 32)), '12': np.zeros((13, 32)), '13': np.zeros((13, 32))}
for row in d_unsh:
    week_data[row[1][-2:]][int(row[1].split('/')[1])][int(row[1].split('/')[0])] += row[-1]
#%%
s = np.sum(week_data['11']) + np.sum(week_data['12']) + np.sum(week_data['13'])
monthly = {'11': np.sum(week_data['11'], axis=1)[1:],
           '12': np.sum(week_data['12'], axis=1)[1:],
           '13': np.sum(week_data['13'], axis=1)[1:]}
monthly['13'] = np.where(monthly['13'] == 0, np.nan, monthly['13'])
#%%
import matplotlib.pyplot as plt
mts = list(range(1, 13))
plt.plot(mts, monthly['11'])
plt.plot(mts, monthly['12'])
plt.plot(mts, monthly['13'])
plt.title('Monthly units sold')
plt.grid()
plt.show()
