#%%
from base_src import Embedding_dataset


DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv'
OUT_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/embedding'
COL = [2, 3]

data_train = Embedding_dataset(DATA_PATH, COL, True, out_path=OUT_PATH)
data_val = Embedding_dataset(DATA_PATH, COL, False, label_encoders=data_train.label_encoders)
