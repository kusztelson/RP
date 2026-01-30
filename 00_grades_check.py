# %%
import pyreadstat
import pandas as pd
import os
from Utils import get_data_path
# Load the .sav file
df, meta = pyreadstat.read_sav(os.path.join(get_data_path(), "INT_01_ST_(2021.04.14)_Public.sav"))


print(len(df))
print(df['Username_Std'].nunique())

columns_to_check = ['Sgrade_Math', 'Sgrade_Read_Lang']
filtered_df = df.dropna(subset=columns_to_check)

print(len(filtered_df))
print(filtered_df['Username_Std'].nunique())
# %%

filtered_df['Username_Std'].to_csv('data_full_data_withoutArts.csv', index=False)
# %%

