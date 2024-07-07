from src import get_args
from src import DataSet

if __name__ == '__main__':
    info = get_args()
    data_mnager = DataSet(paths=info['DATA_PATH'])
#%%