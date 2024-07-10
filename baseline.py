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
plt.plot(mts, monthly['11'], label='yr 1')
plt.plot(mts, monthly['12'], label='yr 2')
plt.plot(mts, monthly['13'], label='yr3')
plt.title('Monthly units sold')
plt.legend()
plt.grid()
plt.show()
#%%
items = [x[3] for x in d_unsh]
items = list(set(items))
item_week = {}
for item in items:
    item_week[item] = np.zeros((3, 13, 32))

for row in d_unsh:
    year = int(row[1].split('/')[2]) - 11
    item_week[row[3]][year][int(row[1].split('/')[1])][int(row[1].split('/')[0])] += row[-1]
#%%
for key in item_week.keys():
    item_week[key] = np.sum(item_week[key], axis=2)[:, 1:]

#%%
fig, axs = plt.subplots(3, 1, constrained_layout=True, figsize=(7, 10))

for key in item_week.keys():
    axs[0].plot(mts, item_week[key][0, :])
    axs[1].plot(mts, item_week[key][1, :])
    axs[2].plot(mts, item_week[key][2, :])

fig.suptitle('Monthly item sold')
axs[0].set_title('yr 1')
axs[1].set_title('yr 2')
axs[2].set_title('yr 3')
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[2].set_yscale('log')
plt.grid()
plt.show()
#%%
item_total = {}
for item in items:
    item_total[item] = 0
for key in item_week.keys():
    item_total[key] = np.sum(item_week[key])
print(item_total)
#%%
x = [str(x) for x in list(item_total.keys())]
y = list(item_total.values())
plt.bar(x, y)
plt.xticks(rotation=45)
plt.show()