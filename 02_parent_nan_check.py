import os

import pandas as pd
import numpy as np
import pyreadstat

from Utils import get_data_path


def get_parent_missing_codes(col_name):
    """
    Returns the list of raw missing value codes based on the INT_02_PA Codebook.
    """
    # --- RULE 1: Occupational Codes (4-digit) ---
    # Source: Page 3 (ISCO)
    # Applies to: MISCO_PA (Mother ISCO), FISCO_PA (Father ISCO)
    if col_name in ['MISCO_PA', 'FISCO_PA']:
        return [9996, 9997, 9999]

    # --- RULE 2: Occupational Indices (2-digit) ---
    # Source: Page 3 (ISEI)
    if col_name in ['MISEI_PA', 'FISEI_PA']:
        return [99]

    # --- RULE 3: Session Metadata ---
    # Source: Page 2
    if col_name == 'Session_LengthPA':
        return [999999]

    # --- RULE 4: Skills (PAA) & Behaviour (PAB) ---
    # Source: Page 4-9 (Skills), Page 10 (Behaviour)
    # Valid: 1-5. Missing: 7 (N/A), 8 (Missing), 9 (Omitted)
    if col_name.startswith('PAA_') or col_name.startswith('PAB'):
        return [7, 8, 9]

    # --- RULE 5: Questionnaire (PAQM) ---
    if col_name.startswith('PAQM'):
        # 5a. Student Birth Year (4-digit)
        # Source: Page 11
        if col_name == 'PAQM00102':
            return [9997, 9998, 9999]

        # 5b. Specific 2-digit Variables
        # These define missing as 97, 98, 99.
        # Source: Page 11 (Birth Month, Parent Age, People in home), Page 12 (Edu), Page 13 (Start ISCED), Page 15 (Life Sat), Page 17 (Expectation)
        two_digit_vars = [
            'PAQM00101', # Student Birth Month
            'PAQM00301', 'PAQM00302', # Parent Age
            'PAQM00401', 'PAQM00402', # Parents living in home
            'PAQM00403', 'PAQM00404', # Siblings in home (Check specifically as they border the change)
            'PAQM00601', 'PAQM00602', # Education Level
            'PAQM01601', # Age start ISCED 0
            'PAQM01701', # Age start ISCED 1
            'PAQM02401', # Life Satisfaction (0-10)
            'PAQM03101'  # Expect child complete ISCED
        ]
        if col_name in two_digit_vars:
            return [97, 98, 99]

        # 5c. All other PAQM variables (1-digit)
        # Includes: Country, Language, Area, Provider, Health, Wellbeing, etc.
        # Default missing: 7, 8, 9
        return [7, 8, 9]

    # --- RULE 6: Weights & Computed Scores ---
    # Source: Page 19-20
    # 4-digit missing codes
    if col_name == 'WT2019_PA' or col_name.endswith('_WLE_ADJ') or col_name == 'ARS_PA':
        return [9999]
    
    # 2-digit missing codes for indices
    if col_name in ['ARS_PA_PAIRS', 'PA_COMM', 'PA_WELLBEING', 'PA_ENCOUR', 'PA_ENGAGE']:
        return [99]

    return []

def check_parent_file_comprehensive(file_path):
    print(f"Loading {file_path}...")
    try:
        df, meta = pyreadstat.read_sav(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Checking {len(df.columns)} columns in Parent dataset...\n")
    
    found_issues = {}

    for col in df.columns:
        codes = get_parent_missing_codes(col)
        if not codes:
            continue
            
        # Check if any of these codes exist in the data
        actual_values = df[col].dropna().unique()
        found_codes = [code for code in codes if code in actual_values]
        
        if found_codes:
            found_issues[col] = found_codes

    # --- REPORTING ---
    print("-" * 60)
    print(f"SUMMARY OF RAW MISSING VALUES FOUND (Parent Data)")
    print("-" * 60)
    
    if len(found_issues) == 0:
        print("SUCCESS: No raw missing values found. Dataset is clean.")
    else:
        print(f"WARNING: Found raw missing values in {len(found_issues)} columns.")
        print("These values must be replaced with NaN before analysis.")
        print("-" * 40)
        print(f"{'Variable':<15} | {'Found Codes'}")
        print("-" * 40)
        for col, codes in found_issues.items():
            print(f"{col:<15} | {codes}")
    
    print("-" * 60)

# Execute
check_parent_file_comprehensive(
    os.path.join(get_data_path(), 'INT_02_PA_(2021.04.14)_Public.sav'))