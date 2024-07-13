import numpy as np
from src import get_args
from src import DataSet

CFG_FILE = '../cfgs/default.yml'
task_info = get_args(CFG_FILE)
Data_manager = DataSet(paths=task_info['DATA_PATH'], year_split=True)
data_train = Data_manager.train_all
data_val = Data_manager.val
#%%
# get stores ids and sku ids
stores = np.unique(data_train[:, 2])
skus = np.unique(data_train[:, 3])
# pick one pair for experiments
store_id = stores[0]
sku_id = skus[0]
#%%
# get all data from selected pair for training and validation
store_match_train = data_train[np.where(data_train[:, 2] == store_id)[0]]
train_single = store_match_train[np.where(store_match_train[:, 3] == sku_id)[0]]

store_match_val = data_val[np.where(data_val[:, 2] == store_id)[0]]
val_single = store_match_val[np.where(store_match_val[:, 3] == sku_id)[0]]

# get only relevant data
train_single = train_single[:, [1, -1]]
val_single = val_single[:, [1, -1]]
#%%
# visualize single series
import matplotlib.pyplot as plt
plt.plot(train_single[:, 0], train_single[:, 1], label='train period')
plt.plot(val_single[:, 0], val_single[:, 1], label='test period')
plt.xticks(rotation=45, fontsize=5)
plt.title(f'Store: {store_id}, sku: {sku_id}')
plt.legend()
plt.show()
#%%
# create autocorrelation plot and get lag_vals
from pandas.plotting import autocorrelation_plot
import pandas as pd

train_series_lag = pd.Series(train_single[:, 1][:val_single.shape[0]].astype(float)) # this time lag cannot exceed val length

autocorrs = [train_series_lag.autocorr(lag=x) for x in list(range(val_single.shape[0]-1))]
autocors_lags = autocorrelation_plot(train_series_lag)
lag_vals = plt.gca().lines[5].get_xydata()[:, 1]
ac_99 = plt.gca().lines[0].get_xydata()[0][1]
plt.title('Autocorrelation plot')
plt.show()
# get most suitable lag
warmup = 2
lag = np.argmin(np.abs(lag_vals[warmup:lag_vals.size-lag_vals.size//2] - ac_99)) + warmup
#%%
# initial ARIMA with no MA (MA = 0)
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas import DataFrame

train_series = pd.Series(train_single[:, 1].astype(float))
train_series.index = pd.Index(pd.to_datetime(train_single[:, 0], dayfirst=True, format="mixed"))
#train_series.index = train_series.index.to_period('M')
val_series = pd.Series(val_single[:, 1].astype(float))
val_series.index = pd.Index(pd.to_datetime(val_single[:, 0], dayfirst=True, format="mixed"))
#val_series.index = val_series.index.to_period('M')

#%%
model = ARIMA(train_series, order=(10,2,1))#, trend='ct')
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())
#%%
# evaluation on val dataset
from math import sqrt

history = [x for x in train_series]
predictions = list()
# walk-forward validation
for t in range(len(val_series)):
    model = ARIMA(history, order=(2, 2, 1))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = val_series[t]
    history.append(obs)
    print(f'predicted={yhat}, expected={obs}')

# evaluate forecasts
rmse = sqrt(mean_squared_error(val_series.values, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.plot(val_series.values)
plt.plot(predictions, color='red')
plt.show()
