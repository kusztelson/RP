import pandas as pd
import numpy as np
import pyreadstat
import os
from Utils import get_data_path

# Files to compare
FILE_ORIG_ST = 'INT_01_ST_(2021.04.14)_Public.sav'
FILE_ORIG_PA = 'INT_02_PA_(2021.04.14)_Public.sav'
FILE_ORIG_TC = 'INT_03_TC_(2021.04.14)_Public.sav'
FILE_NEW     = 'INT_Final_Merged_Prefixed.sav'

def compare_datasets():
    print("--- LOADING DATASETS FOR COMPARISON ---\n")
    try:
        df_st, meta_st = pyreadstat.read_sav(os.path.join(get_data_path(), FILE_ORIG_ST))
        df_pa, meta_pa = pyreadstat.read_sav(os.path.join(get_data_path(), FILE_ORIG_PA))
        df_tc, meta_tc = pyreadstat.read_sav(os.path.join(get_data_path(), FILE_ORIG_TC))
        df_new, meta_new = pyreadstat.read_sav(FILE_NEW)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"{'METRIC':<30} | {'ORIGINAL (ST/PA/TC)':<25} | {'NEW MERGED FILE':<20} | {'STATUS'}")
    print("-" * 90)

    # 1. ROW COUNT CHECK (Did we lose students?)
    # ---------------------------------------------------------
    # The merged file should match the Student file exactly (Left Join)
    st_count = len(df_st)
    new_count = len(df_new)
    status = "MATCH" if st_count == new_count else "MISMATCH"
    print(f"{'Row Count (Students)':<30} | {str(st_count):<25} | {str(new_count):<20} | {status}")

    # 2. COLUMN COUNT CHECK
    # ---------------------------------------------------------
    # New cols should roughly be Sum(Old Cols) - 2 (Joined Keys)
    # This is an approximation because we might have dropped some during cleaning, 
    # but usually it's Sum - duplicates.
    total_orig_cols = len(df_st.columns) + len(df_pa.columns) + len(df_tc.columns)
    new_cols = len(df_new.columns)
    print(f"{'Column Count':<30} | {str(total_orig_cols):<25} | {str(new_cols):<20} | {'Info Only'}")

    print("-" * 90)
    print("DATA CLEANING VERIFICATION (Before vs After)")
    print("-" * 90)

    # 3. CLEANING CHECK: PARENT OCCUPATION (9999)
    # ---------------------------------------------------------
    # Original: 'MISCO_PA' should have 9999
    # New: 'PA_MISCO_PA' should NOT have 9999
    orig_dirty = 9999 in df_pa['MISCO_PA'].values if 'MISCO_PA' in df_pa.columns else "N/A"
    new_dirty  = 9999 in df_new['PA_MISCO_PA'].values if 'PA_MISCO_PA' in df_new.columns else "N/A"
    
    check_status = "PASS" if (orig_dirty is True and new_dirty is False) else "FAIL"
    print(f"{'Check: Dirty Code 9999 (Occ)':<30} | {'Present':<25} | {'Gone (Clean)':<20} | {check_status}")

    # 4. CLEANING CHECK: PARENT LIKERT (8)
    # ---------------------------------------------------------
    # Original: 'PAA_SOC03' should have 8
    # New: 'PA_PAA_SOC03' should NOT have 8
    orig_likert_dirty = 8 in df_pa['PAA_SOC03'].values if 'PAA_SOC03' in df_pa.columns else "N/A"
    new_likert_dirty  = 8 in df_new['PA_PAA_SOC03'].values if 'PA_PAA_SOC03' in df_new.columns else "N/A"
    
    check_status_2 = "PASS" if (orig_likert_dirty is True and new_likert_dirty is False) else "FAIL"
    print(f"{'Check: Dirty Code 8 (Scale)':<30} | {'Present':<25} | {'Gone (Clean)':<20} | {check_status_2}")

    print("-" * 90)
    print("METADATA PRESERVATION (Label Mapping)")
    print("-" * 90)

    # 5. METADATA CHECK
    # ---------------------------------------------------------
    # Verify a Student label
    orig_label = meta_st.column_names_to_labels.get('STQM00101', 'Not Found')
    new_label  = meta_new.column_names_to_labels.get('ST_STQM00101', 'Not Found')
    match = "MATCH" if orig_label == new_label else "MISMATCH"
    print(f"{'Label: Student Grade':<30} | {orig_label[:20]+'...':<25} | {new_label[:20]+'...':<20} | {match}")

    # Verify a Parent label
    orig_label_pa = meta_pa.column_names_to_labels.get('PAQM00601', 'Not Found')
    new_label_pa  = meta_new.column_names_to_labels.get('PA_PAQM00601', 'Not Found')
    match_pa = "MATCH" if orig_label_pa == new_label_pa else "MISMATCH"
    print(f"{'Label: Parent Education':<30} | {orig_label_pa[:20]+'...':<25} | {new_label_pa[:20]+'...':<20} | {match_pa}")

if __name__ == "__main__":
    compare_datasets()