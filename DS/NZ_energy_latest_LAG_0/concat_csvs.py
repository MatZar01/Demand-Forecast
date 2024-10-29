import os
import pandas as pd
import re

# Directory containing the CSV files
directory = '/Users/nuwan.gunasekara/Desktop/CODE/NewZealandEnergyPrices/full_data'

# Regular expression pattern to match filenames in the format all_<PoC>_data.csv
pattern = r'all_(.*?)_data\.csv'

# Empty list to store dataframes
dfs = []

# Loop through all files in the directory
for file_name in os.listdir(os.path.join(directory)):
    # Check if the filename matches the pattern
    match = re.match(pattern, file_name)
    if match:
        # Extract the PoC part from the filename
        poc = match.group(1)

        # Full file path
        file_path = os.path.join(directory, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Add the 'PoC' column as the first column
        df.insert(0, 'PoC', poc)  # Insert 'PoC' at index 0

        # Append the DataFrame to the list
        dfs.append(df)

# Print the list of DataFrames
print(f"Loaded {len(dfs)} files into the list.")

# Concatenate all the DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

combined_df.drop(columns=['Med$PerMWHr'], inplace=True)
combined_df = combined_df[['PoC','TradingPeriod','IsProxyPriceFlag','Year','Month','Day','PrevAvg$PerMWHr','PrevMed$PerMWHr','SinPeriod','CosPeriod','SinDate','CosDate', 'Avg$PerMWHr']]

# Year, Month, Day, Trading period
combined_df.sort_values(by=['Year', 'Month', 'Day', 'TradingPeriod'], ascending=[True, True, True, True], inplace=True)

# Save the concatenated DataFrame to a new CSV file
output_file = os.path.join('./', 'combined_data.csv')
combined_df.to_csv(output_file, index=False)

print(f"Combined dataframe saved to {output_file}")
