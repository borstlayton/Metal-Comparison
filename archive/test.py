import pandas as pd
import matplotlib.pyplot as plt

# File paths
CSV_FILE_2000 = "assets/metals_2000.csv"
CSV_FILE_2025 = "assets/metals_2025.csv"

# Load csv files, skipping first two rows
df_2000 = pd.read_csv(CSV_FILE_2000)
df_2025 = pd.read_csv(CSV_FILE_2025)

print(df_2000.head())

# Remove the units row from dataframes
# df_2000.drop(index=0)
# df_2025.drop(index=0)

# Drop 'Sample ID'
# df_2000 = df_2000.drop(columns=['Sample ID'])
# df_2025 = df_2025.drop(columns=['Sample ID'])

# print(df_2000.head())
# print(df_2025.head())





