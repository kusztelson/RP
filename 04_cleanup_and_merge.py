import datetime

import pandas as pd
import numpy as np
import pyreadstat
import os
from Utils import get_data_path

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("Loading files...")
prefix = get_data_path()
df_st, meta_st = pyreadstat.read_sav(os.path.join(prefix, 'INT_01_ST_(2021.04.14)_Public.sav'))
df_pa, meta_pa = pyreadstat.read_sav(os.path.join(prefix, 'INT_02_PA_(2021.04.14)_Public.sav'))
df_tc, meta_tc = pyreadstat.read_sav(os.path.join(prefix, 'INT_03_TC_(2021.04.14)_Public.sav'))
df_tcq, meta_tcq = pyreadstat.read_sav(os.path.join(prefix, 'INT_04_TCQ_(2021.04.14)_Public.sav'))
df_pr, meta_pr = pyreadstat.read_sav(os.path.join(prefix, 'INT_05_PRQ_(2021.04.14)_Public.sav'))

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

pr_group_options = {
    "FullID_PR": "first",
    "Username_PR": "first",
    "CohortID": "first",
    "SiteID": "first",
    "SchID": "first",
    "SampleStatus": "first",
    "MultiCohort": "first",
    "LANG_PR": "first",
    "Start_date_PR": "first",
    "Start_time_PR": "first",
    "End_date_PR": "first",
    "End_time_PR": "first",
    "Session_LengthPR": "first",
    "PRQM00601": "median",
    "PRQM00801": "median",
    "PRQM00802": "median",
    "PRQM00803": "median",
    "PRQM00804": "median",
    "PRQM00805": "median",
    "PRQM00901": "median",
    "PRQM01001": "median",
    "PRQM01002": "median",
    "PRQM01003": "median",
    "PRQM01004": "median",
    "PRQM01005": "median",
    "PRQM01006": "median",
    "PRQM01007": "median",
    "PRQM01008": "median",
    "PRQM01009": "median",
    "PRQM01010": "median",
    "PRQM01101": "median",
    "PRQM01102": "median",
    "PRQM01103": "median",
    "PRQM01104": "median",
    "PRQM01201": "median",
    "PRQM01301": "median",
    "PRQM01302": "median",
    "PRQM01401": "median",
    "PRQM01402": "median",
    "PRQM01403": "median",
    "PRQM01404": "median",
    "PRQM01405": "median",
    "PRQM01406": "median",
    "PRQM01407": "median",
    "PRQM01408": "median",
    "PRQM01501": "median",
    "PRQM01601": "median",
    "PRQM01602": "median",
    "PRQM01603": "median",
    "PRQM01604": "median",
    "PRQM01701": "median",
    "PRQM01702": "median",
    "PRQM01703": "median",
    "PRQM01801": "median",
    "PRQM01802": "median",
    "PRQM01803": "median",
    "PRQM01804": "median",
    "PRQM01805": "median",
    "PRQM01806": "median",
    "PRQM01901": "median",
    "PRQM01902": "median",
    "PRQM01903": "median",
    "PRQM01904": "median",
    "PRQM01905": "median",
    "PRQM01906": "median",
    "PRQM01907": "median",
    "PRQM01908": "median",
    "PRQM01909": "median",
    "PRQM02001": "median",
    "PRQM02101": "median",
    "PRQM02102": "median",
    "PRQM02103": "median",
    "PRQM02104": "median",
    "PRQM02105": "median",
    "PRQM02106": "median",
    "PRQM02107": "median",
    "PRQM02108": "median",
    "PRQM02109": "median",
    "PRQM02110": "median",
    "PRQM02111": "median",
    "PRQM02201": "median",
    "PRQM02202": "median",
    "PRQM02203": "median",
    "PRQM02204": "median",
    "PRQM02205": "median",
    "PRQM02301": "median",
    "PRQM02401": "median",
    "PRQM02402": "median",
    "PRQM02403": "median",
    "PRQM02404": "median",
    "PRQM02405": "median",
    "PRQM02406": "median",
    "PRQM02501": "median",
    "PRQM02502": "median",
    "PRQM02503": "median",
    "PRQM02504": "median",
    "PRQM02505": "median",
    "PRQM02506": "median",
    "PRQM02507": "median",
    "PRQM02601": "median",
    "PRQM02602": "median",
    "PRQM02603": "median",
    "PRQM02604": "median",
    "PRQM02605": "median",
    "PRQM02606": "median",
    "PRQM02607": "median",
    "PRQM02701": "median",
    "PRQM02702": "median",
    "PRQM02703": "median",
    "PRQM02704": "median",
    "PRQM02705": "median",
    "PRQM02706": "median",
    "PRQM02801": "median",
    "PRQM02901": "median",
    "PRQM02902": "median",
    "PRQM02903": "median",
    "PRQM02904": "median",
    "PRQM03001": "median",
    "PRQM03002": "median",
    "PRQM03003": "median",
    "PRQM03004": "median",
    "PRQM03005": "median",
    "PRQM03006": "median",
    "PRQM03101": "median",
    "PRQM03201": "median",
    "WT2019_PR": "median",
    "pr_promsses": "median",
    "pr_stubeha": "median",
    "pr_teabeha": "median"
}
df_pr = (df_pr.groupby("SchID", as_index=False)
         .agg(pr_group_options))

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

# Apply to Teacher (TCQ_)
# Note: Teacher key is 'Username_Std' in file (with underscore)
df_tcq_new = apply_prefix_and_extract_meta(df_tcq, meta_tcq, 'TCQ')

# Apply to Principal Questioner (PR_)
# Note: School key is 'SchID' in file
df_pr_new = apply_prefix_and_extract_meta(df_pr, meta_pr, 'PR')

# ---------------------------------------------------------
# 4. MERGE (On standardized key)
# ---------------------------------------------------------
print("Merging datasets...")

# Merge Student + Parent
df_merged = pd.merge(df_st_new, df_pa_new, on='Username_Std', how='left')

# Merge Result + Teacher
df_merged = pd.merge(df_merged, df_tc_new, on='Username_Std', how='left')

# Merge Result + Teacher Self Questioner
df_merged = pd.merge(df_merged, df_tcq_new, left_on='ST_Username_TC', right_on='TCQ_Username_TC', how='left')

# Merge Result + Principal
df_merged = pd.merge(df_merged, df_pr_new, left_on='ST_SchID', right_on='PR_SchID', how='left')

print("Fixing time and date objects if exist")
for col in df_merged.columns:
    if df_merged[col].apply(lambda x: isinstance(x, datetime.date)).any():
        df_merged[col] = pd.to_datetime(df_merged[col])

for col in df_merged.columns:
    if df_merged[col].apply(lambda x: isinstance(x, datetime.time)).any():
        df_merged[col] = df_merged[col].astype(str)

# ---------------------------------------------------------
# 5. SAVE
# ---------------------------------------------------------

print("Saving to INT_Final_Merged_Prefixed.sav...")
pyreadstat.write_sav(
    df_merged,
    "INT_Final_Merged_Prefixed.sav",
    column_labels=final_col_labels,
    variable_value_labels=final_val_labels
)

print(f"Success! Saved {df_merged.shape[0]} rows.")
print("Example columns: ", list(df_merged.columns[:5]))