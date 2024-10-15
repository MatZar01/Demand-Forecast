import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# Function to create tumbling windows
def compute_tumbling_windows(data_frame, window_size):
    # Reshape data into non-overlapping windows
    windows = np.array_split(data_frame, np.arange(window_size, len(data_frame), window_size))
    return windows

# Create the parser
parser = argparse.ArgumentParser()
# Add a string argument
parser.add_argument('-d','--dir', type=str, help="file", default='../DS/demand-forecasting_LAG_15')
# Parse the arguments
args = parser.parse_args()


with open(os.path.join(args.dir, 'config.json') , 'r') as file:
    config = json.load(file)


with open(os.path.join(args.dir, 'results.pkl'), 'rb') as f:
    results = pickle.load(f)


WINDOW_SIZE = results['WINDOW_SIZE']
WINDOW_SIZE = 1000
# drift_detectors = results['drift_detectors']
incremental_learners = results['incremental_learners']
true_value = results['true_value']
y_hat = results['y_hat']
y_hat_none_at = results['y_hat_none_at']
# 'y_hat_nan_at': np.where(np.isnan(np.array(y_hat)))[0],
y_hat_nan_at = results['y_hat_nan_at']
idxs = results['idxs']
RANDOM_TRAIN_INC = results['RANDOM_TRAIN_INC']
evaluator_results = results['evaluator_results']


# Define window sizes
plot_data = {
    'true_value': [true_value, None],
    'MLP(embeddings)': [y_hat, evaluator_results]
}

for l_name,l in incremental_learners.items():
    post_fix = 'Random Train'if RANDOM_TRAIN_INC else ''
    plot_data[f'{l_name} {post_fix}'] = [l['y_hat'], l['evaluator_results']]


# Create a figure with subplots
fig, axes = plt.subplots(len(plot_data), 1, figsize=(10, 10), sharex=True)

# Plot each window size
true_window_centers = None
true_means = None
true_stds = None
i = 0
for k, v in plot_data.items():
    df = pd.DataFrame(v[0])
    # Compute means and stds for each window
    means = np.array([window.mean().item() for window in compute_tumbling_windows(df, WINDOW_SIZE)])
    stds = np.array([window.std().item() for window in compute_tumbling_windows(df, WINDOW_SIZE)])

    # Create an index for the windows (center points of each window)
    window_centers = np.arange(WINDOW_SIZE // 2, len(df), WINDOW_SIZE)
    if len (window_centers) < len(means):
        window_centers = np.append(window_centers, window_centers[-1] + (len(df)-window_centers[-1]) // 2)
    if k == 'true_value':
        true_window_centers = window_centers
        true_means = means
        true_stds = stds

    # axes[i].plot(df, label=f'{k}', color='blue', alpha=0.1)
    axes[i].plot(window_centers, means, label=f'Mean (window={WINDOW_SIZE})', color='red' if k == 'true_value' else 'green')
    axes[i].fill_between(window_centers, means - stds, means + stds, color='lightpink' if k == 'true_value' else 'lightgreen', alpha=0.3, label='tumbling Std Dev')

    axes[i].plot(true_window_centers, true_means, label=f'true Mean (window={WINDOW_SIZE})', color='red')
    axes[i].fill_between(true_window_centers, true_means - true_stds, true_means + true_stds, color='lightpink', alpha=0.3, label='true tumbling Std Dev')

    # Add labels and legend
    results_str = "" if v[1] is None else str(v[1]).replace('{', '').replace('}', '').split(',')[-1]
    axes[i].set_title(f'{k}\n{results_str}')
    axes[i].set_ylabel('Value')
    axes[i].legend()
    i += 1

# Set common labels and display
axes[-1].set_xlabel('Index')
plt.tight_layout()
plt.savefig(os.path.join(args.dir, 'plot.png'))
plt.show()




