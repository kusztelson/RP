import pandas as pd
import numpy as np
import pyreadstat

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("Loading files...")
df_st, meta_st = pyreadstat.read_sav('INT_01_ST_(2021.04.14)_Public.sav')
df_pa, meta_pa = pyreadstat.read_sav('INT_02_PA_(2021.04.14)_Public.sav')
df_tc, meta_tc = pyreadstat.read_sav('INT_03_TC_(2021.04.14)_Public.sav')

# ---------------------------------------------------------
# 2. CLEANING (Applying Codebook Rules)
# ---------------------------------------------------------
print("Cleaning data...")

# --- Clean Parent ---
# Occupational Codes
for col in ['MISCO_PA', 'FISCO_PA']:
    if col in df_pa.columns:
        df_pa[col] = df_pa[col].replace({9996: np.nan, 9999: np.nan})

# Likert Scales (Missing by design '8')
dirty_likert_cols = ['PAA_COO03', 'PAA_CUR04', 'PAA_EMO03', 'PAA_EMP03', 'PAA_ENE03', 
                     'PAA_OPT04', 'PAA_SOC03', 'PAA_STR02', 'PAA_TOL03', 'PAA_TRU03']
for col in dirty_likert_cols:
    if col in df_pa.columns:
        df_pa[col] = df_pa[col].replace({8: np.nan})

# --- Clean Student ---
# Occupational Codes (Safety)
for col in ['MISCO_ST', 'FISCO_ST', 'STISCO_ST']:
    if col in df_st.columns:
        df_st[col] = df_st[col].replace({9996: np.nan, 9999: np.nan})

# --- Clean Teacher ---
# Mean Scores
for col in df_tc.columns:
    if col.endswith('_TC_Mean') or col.endswith('_TC_mean'):
        df_tc[col] = df_tc[col].replace({9999: np.nan})

# ---------------------------------------------------------
# 3. PREFIXING COLUMNS & UPDATING METADATA
# ---------------------------------------------------------
print("Applying prefixes and mapping metadata...")

# Global Metadata containers for the final file
final_col_labels = {}
final_val_labels = {}

def apply_prefix_and_extract_meta(df, meta, prefix):
    """
    1. Renames all columns (except key) with prefix.
    2. Updates metadata to match new names.
    3. Returns new DF.
    """
    new_df = df.copy()
    rename_map = {}
    
    for col in df.columns:
        # SKIP the merge key (keep it standard for merging)
        if col == 'Username_Std':
            rename_map[col] = col # No change
            continue
            
        # Define New Name
        new_name = f"{prefix}_{col}"
        rename_map[col] = new_name
        
        # 1. Update Column Label (The Question)
        if col in meta.column_names_to_labels:
            final_col_labels[new_name] = meta.column_names_to_labels[col]
            
        # 2. Update Value Labels (The Answers)
        # pyreadstat.write_sav expects: { 'col_name': {1: 'Male', 2: 'Female'} }
        # We must look up the label set name first, then get the dict
        if col in meta.variable_to_label:
            label_set_name = meta.variable_to_label[col]
            if label_set_name in meta.variable_value_labels:
                final_val_labels[new_name] = meta.variable_value_labels[label_set_name]

    # Apply renaming to dataframe
    new_df = new_df.rename(columns=rename_map)
    return new_df

# Apply to Student (ST_)
df_st_new = apply_prefix_and_extract_meta(df_st, meta_st, 'ST')

# Apply to Parent (PA_)
df_pa_new = apply_prefix_and_extract_meta(df_pa, meta_pa, 'PA')

# Apply to Teacher (TC_)
# Note: Teacher key is 'Username_Std' in file (with underscore)
df_tc_new = apply_prefix_and_extract_meta(df_tc, meta_tc, 'TC')

# ---------------------------------------------------------
# 4. MERGE (On standardized key)
# ---------------------------------------------------------
print("Merging datasets...")

# Merge Student + Parent
df_merged = pd.merge(df_st_new, df_pa_new, on='Username_Std', how='left')

# Merge Result + Teacher
df_merged = pd.merge(df_merged, df_tc_new, on='Username_Std', how='left')

# ---------------------------------------------------------
# 5. SAVE
# ---------------------------------------------------------
print("Saving to INT_Final_Merged_Prefixed.sav ...")

pyreadstat.write_sav(
    df_merged,
    "INT_Final_Merged_Prefixed.sav",
    column_labels=final_col_labels,
    variable_value_labels=final_val_labels
)

print(f"Success! Saved {df_merged.shape[0]} rows.")
print("Example columns: ", list(df_merged.columns[:5]))