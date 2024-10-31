import os
import pandas as pd
import re

# Directory containing the CSV files
directory = './'

# Full file path
file_path = os.path.join(directory, 'train.csv')

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert 'week' to datetime format
df['week'] = pd.to_datetime(df['week'], format='%d/%m/%y')

# Sort by week
df.sort_values(by=['week'], ascending=True, inplace=True)

output_file = os.path.join('./', 'sorted_train.csv')
df.to_csv(output_file, index=False, date_format='%d/%m/%y')

# Shift 'units sold' by one row within each group of 'store_id' and 'SKU_id'
df['previous_units_sold'] = df.groupby(['store_id', 'sku_id'])['units_sold'].shift(1)

# Fill NaN values in the first row of each group with 0 (if you want)
df['previous_units_sold'].fillna(0, inplace=True)

columns = list(df.columns)
columns.remove('units_sold')
columns.append('units_sold')

# Save the DataFrame to a new CSV file with the exact date format
output_file = os.path.join('./', 'sorted_shifted_train.csv')
df[columns].to_csv(output_file, index=False, date_format='%d/%m/%y')

print(f"Updated DataFrame saved to {output_file}")
