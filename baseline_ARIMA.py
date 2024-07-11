import numpy as np
from src import get_args
from src import DataSet

CFG_FILE = 'cfgs/default.yml'
task_info = get_args(CFG_FILE)
Data_manager = DataSet(paths=task_info['DATA_PATH'], year_split=True)
data_train = Data_manager.train_all[:, 1:]
data_val = Data_manager.val[:, 1:]
#%%