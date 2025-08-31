import pandas as pd
import matplotlib.pyplot as plt

# Load Excel files, skipping first two rows
df_2000 = pd.read_csv("metals_2000.csv", skiprows=[1,2])
df_2025 = pd.read_csv("metals_2025.csv", skiprows=[1,2])

# print(df_2000.head())

# Load units row
units_2000 = pd.read_csv("metals_2000.csv").iloc[0].to_dict()
units_2025 = pd.read_csv("metals_2025.csv").iloc[0].to_dict()

# print(units_2000)

# Remove the units row from dataframes
# df_2000.drop(index=0)
# df_2025.drop(index=0)

# Drop 'Sample ID'
df_2000 = df_2000.drop(columns=['Sample ID'])
df_2025 = df_2025.drop(columns=['Sample ID'])

print(df_2000.head())
print(df_2025.head())





