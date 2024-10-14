import pickle

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

file_path = 'combined_data.csv'
df = pd.read_csv(file_path)

one_hot_info = [
    ['PoC', 'C0'],
    ['TradingPeriod', 'C1']
]

for h in one_hot_info:
    # Instantiate the encoder
    encoder = OneHotEncoder()

    # Fit
    encoder.fit(df[[h[0]]].to_numpy())

    # Save the encoder to a pickle file
    pkl_file_name = f'onehot_{h[1]}.pkl'
    with open(pkl_file_name, 'wb') as file:
        pickle.dump(encoder, file)

    print(f"Encoder saved to {pkl_file_name}")




