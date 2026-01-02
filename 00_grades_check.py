# %%
import pyreadstat
import pandas as pd
# Load the .sav file
df, meta = pyreadstat.read_sav("INT_01_ST_(2021.04.14)_Public.sav")


print(len(df))
print(df['Username_Std'].nunique())

columns_to_check = ['Sgrade_Math', 'Sgrade_Read_Lang', 'Sgrade_Arts']
filtered_df = df.dropna(subset=columns_to_check)

print(len(filtered_df))
print(filtered_df['Username_Std'].nunique())
# %%

filtered_df['Username_Std'].to_csv('data_full_data.csv', index=False)
# %%

