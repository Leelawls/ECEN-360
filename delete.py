# Updated Cell: Correlation Matrix Including ERCOT.LOAD, Excluding Date

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

# 1) If your df_solar DataFrame includes 'Date' (or any other non-numeric columns),
#    we can drop them. If 'Date' isn't in df_solar, the 'errors="ignore"' just skips it.

df_solar_subset = pd.read_csv("Load.csv", index_col=0).drop(
    columns=["Date"], errors="ignore"
)

# 2) Keep only numeric columns (float, int)
df_solar_subset = df_solar_subset.select_dtypes(include=["float", "int"])

# 3) Compute correlation matrix (now includes ERCOT.LOAD, if present)
corr_solar = df_solar_subset.corr()

print("Correlation Matrix (Including ERCOT.LOAD, Excluding Date):")
print(corr_solar)

# 4) Visualize as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_solar, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap (Including ERCOT.LOAD, Excluding Date)")
plt.show()
