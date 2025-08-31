import pandas as pd
import matplotlib.pyplot as plt

# Load Excel files, skipping first two rows
df_2000 = pd.read_csv("metals_2000.csv", skiprows=[1,2])
df_2025 = pd.read_csv("metals_2025.csv", skiprows=[1,2])

# Load units row
units_2000 = pd.read_csv("metals_2000.csv").iloc[0].to_dict()
units_2025 = pd.read_csv("metals_2025.csv").iloc[0].to_dict()

# Drop 'Sample ID' column if present
df_2000 = df_2000.drop(columns=['Sample ID'])
df_2025 = df_2025.drop(columns=['Sample ID'])

# Compute mean concentrations for each metal
mean_2000 = df_2000.mean()
mean_2025 = df_2025.mean()

# Find common metals
common_metals = mean_2000.index.intersection(mean_2025.index)

# Prepare data for plotting
x = range(len(common_metals))
width = 0.35
values_2000 = mean_2000[common_metals]
values_2025 = mean_2025[common_metals]
unit_label = units_2000.get(common_metals[0], 'mg/kg')  # fallback unit

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([i - width/2 for i in x], values_2000, width=width, label='2000')
ax.bar([i + width/2 for i in x], values_2025, width=width, label='2025')

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(common_metals, rotation=45)
ax.set_ylabel(f'Concentration {unit_label}')
ax.set_title('Average Metal Concentrations: 2000 vs 2025')
ax.legend()
plt.tight_layout()
plt.show()