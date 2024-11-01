
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

file_path = 'combined_data.csv'
df = pd.read_csv(file_path)


for i, c in enumerate(df.columns):
    print(f'{i} {c}')

print (f'{len(df.columns)}')

print('[', end='')
for i, c in enumerate(df.columns):
    print(f'{i}', end=',')
print(']')

print('[', end='')
for i, c in enumerate(df.columns):
    print(f"'{c}'", end=',')
print(']')

for c in ['PoC', 'TradingPeriod']:
    print(f'{c} {len(df[c].unique())} {df[c].unique()}')


# Group by 'PoC' and 'TradingPeriod' and count occurrences
combination_counts = df.groupby(['PoC', 'TradingPeriod']).size().reset_index(name='Count').sort_values(by='Count', ascending=False)

# Display the result
print(combination_counts)
#          PoC  TradingPeriod  Count
# 0    ALB0331              1    546
# 193  SDN0331             44    546
# 202  STK0331              3    546
# 201  STK0331              2    546
# 200  STK0331              1    546
# ..       ...            ...    ...
# 198  SDN0331             49      2
# 199  SDN0331             50      2
# 248  STK0331             49      2
# 249  STK0331             50      2
# 349  WIL0331             50      2
print(combination_counts['Count'].median())
# 546.0

print(combination_counts[combination_counts['Count'] < 527])

# ALB0331,49,2
# ALB0331,50,2
# WIL0331,49,2
# WGN0331,50,2
# ISL0661,50,2
# HAM0331,49,2
# HAM0331,50,2
# ISL0661,49,2
# WGN0331,49,2
# SDN0331,49,2
# SDN0331,50,2
# STK0331,49,2
# STK0331,50,2
# WIL0331,50,2