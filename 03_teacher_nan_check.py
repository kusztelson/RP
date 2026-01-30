import os

import pandas as pd
import numpy as np
import pyreadstat

from Utils import get_data_path


def get_teacher_missing_codes(col_name):
    """
    Returns the list of raw missing value codes based on the INT_03_TC Codebook.
    """
    # Rule 1: Session Metadata (Page 2)
    if col_name == 'Session_LengthTCA':
        return [999999]

    # Rule 2: Teacher INA Items (Contact/Demographics) (Page 3)
    if col_name.startswith('INA'):
        return [7, 8, 9]

    # Rule 3: Teacher Indirect Assessment - Skills (Page 4-6)
    if col_name.startswith('TCA_'):
        return [7, 8, 9]

    # Rule 4: Teacher Behavioural Indicators (Page 7)
    if col_name.startswith('TCB'):
        return [7, 8, 9]
    
    # Rule 5: Student Academic Performance (Page 8)
    if col_name.startswith('TCQM'):
        return [7, 8, 9]

    # Rule 6: Mean Scores (Page 8)
    # Note: Handles both 'Mean' and 'mean' (e.g., STR_TC_mean is lowercase in codebook)
    if col_name.endswith('_TC_Mean') or col_name.endswith('_TC_mean'):
        return [9999]

    return []

def check_teacher_file(file_path):
    print(f"Loading {file_path}...")
    try:
        df, meta = pyreadstat.read_sav(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Checking {len(df.columns)} columns in Teacher dataset...\n")
    
    found_issues = {}

    for col in df.columns:
        codes = get_teacher_missing_codes(col)
        if not codes:
            continue
            
        # Check if any of these codes exist in the data
        actual_values = df[col].dropna().unique()
        found_codes = [code for code in codes if code in actual_values]
        
        if found_codes:
            found_issues[col] = found_codes

    # --- REPORTING ---
    print("-" * 60)
    print(f"SUMMARY OF RAW MISSING VALUES FOUND (Teacher Data)")
    print("-" * 60)
    
    if len(found_issues) == 0:
        print("SUCCESS: No raw missing values found. Dataset is clean.")
    else:
        print(f"WARNING: Found raw missing values in {len(found_issues)} columns.")
        print("These values must be replaced with NaN before analysis.")
        print("-" * 40)
        print(f"{'Variable':<20} | {'Found Codes'}")
        print("-" * 40)
        for col, codes in found_issues.items():
            print(f"{col:<20} | {codes}")
    
    print("-" * 60)

# Run the check
check_teacher_file(
    os.path.join(get_data_path(), 'INT_03_TC_(2021.04.14)_Public.sav'))