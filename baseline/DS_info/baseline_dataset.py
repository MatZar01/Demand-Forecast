#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.loadtxt('/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv', skiprows=1, delimiter=',', dtype=str)
dates = data[:, 1]
data = data[:, 2:].astype(float)

uniques = np.unique(data[:, 0])

outs = []
out_dates = []

for num in uniques:
    match = np.where(data[:, 0]==num)
    sales = data[:, -1][match]
    date = dates[match]
    d_out = []
    for d in date:
        d_out.append(np.datetime64(f'{int(d.split("/")[-1])+2000}-{d.split("/")[1]}-{d.split("/")[0]}'))
    d_out = np.array(d_out)

    out_dates.append(d_out)
    outs.append(sales)

out_by_month = []
for i in range(len(outs)):
    df = pd.DataFrame({'dates': out_dates[i], 'sales': outs[i]})

    df['dates'] = pd.to_datetime(df['dates'])

    df.set_index('dates', inplace=True)

    monthly_sales = df.resample('ME').sum()
    monthly_sales = monthly_sales['sales'].to_numpy()

    out_by_month.append(monthly_sales)

#%%
plt.figure(figsize=(8,5))
plt.plot(np.array(out_by_month).T, label=uniques.astype(int), alpha=0.8)
plt.yscale('log')
plt.grid(True)
plt.xlabel('Time [months]', fontsize=14)
plt.ylabel('Units sold [log scale]', fontsize=14)
plt.title('Units sold monthly by SKU', fontsize=18)
plt.axvline(x=24, color='r', linestyle='--', linewidth=2)

ax = plt.axes([0.4, 0.13, 0.3, 0.08])
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor((1, 1, 1, 0.5))
text_box = ax.text(0.5, 0.0, "Each line color represents\n one of 76 unique SKUs", ha='center', va='bottom', fontsize=13)

ax2 = plt.axes([0.125, 0.88, 0.65, 0.04])
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_facecolor((1, 1, 1, 0.5))
text_box2 = ax2.text(0.5, 0.0, "Training data", ha='center', va='bottom', fontsize=13)

ax2 = plt.axes([0.795, 0.88, 0.19, 0.04])
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_facecolor((1, 1, 1, 0.5))
text_box2 = ax2.text(0.5, 0.0, "Evaluation data", ha='center', va='bottom', fontsize=13)

plt.tight_layout(pad=0.5)
plt.show()
#%%
vars = [np.std(s) for s in outs]
