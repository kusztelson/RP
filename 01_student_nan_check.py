import os

import pandas as pd
import numpy as np
import pyreadstat

from Utils import get_data_path


def get_student_missing_codes(col_name):
    """
    Returns the list of raw missing value codes based on the Student Codebook.
    """
    # Rule 1: Occupational Codes (4-digit) & Session
    # Source: Page 3 (ISCO), Page 2 (Session)
    if col_name in ['MISCO_ST', 'FISCO_ST', 'STISCO_ST']:
        return [9996, 9997, 9999]
    if col_name == 'Session_LengthS':
        return [999999]

    # Rule 2: Occupational Indices & Derived Scores (2-digit)
    # Source: Page 3 (ISEI), Page 21 (Indices)
    if col_name in ['MISEI_ST', 'FISEI_ST', 'STISEI_ST']:
        return [99]
    if col_name.startswith('ST_') and col_name not in ['ST_RELTEACH', 'ST_BULLY']: # Catch-all for ST_ indices usually 99
        return [99]
    if col_name == 'ARS_PAIRS':
        return [99]
    if col_name == 'IMMBACK':
        return [9]

    # Rule 3: Skills (STA) & Behavioural (STB) - 1-digit
    # Source: Page 4 (Skills), Page 10 (Behavioural)
    # Codes: 7 (N/A), 8 (Missing by design), 9 (Omitted)
    if col_name.startswith('STA_') or col_name.startswith('STB'):
        return [7, 8, 9]

    # Refined Rule 4: Cognitive Items (COGM)
    if col_name.startswith('COGM'):
        # EXCEPTION: COGM00401 (Number of fish)
        # 7 is VALID here. Only 98 is missing.
        if col_name == 'COGM00401':
            return [98]
        
        # COGM Items (Raw responses usually use 2-digit missing codes)
        # Examples: COGM00301 (Height), COGM00501 (Day)
        # These use 97, 98, 99. They do NOT use 7, 8, 9 as missing.
        if not col_name.endswith('S') and not col_name.endswith('5'):
             return [97, 98, 99]

        # COGM Scores (Usually end in 'S' or '5')
        # Examples: COGM00401S, COGM003015
        # These use 7, 8, 9 as missing.
        return [7, 8, 9]

    # Rule 5: Questionnaire (STQM)
    # Source: Pages 12-20
    if col_name.startswith('STQM'):
        # Sub-rule 5a: 4-digit missing codes (Year, Height, Weight)
        four_digit_vars = [
            'STQM00302', # Birth Year
            'STQM00501', # Height
            'STQM00601'  # Weight
        ]
        if col_name in four_digit_vars:
            return [9997, 9998, 9999]

        # Sub-rule 5b: 2-digit missing codes
        # Grade, Time at school, Birth Month, Age started ISCED, Life Satisfaction
        two_digit_vars = [
            'STQM00101', # Grade
            'STQM00201', # Time at school
            'STQM00301', # Birth Month
            'STQM01701', # Age started ISCED 0
            'STQM01801', # Age started ISCED 1
            'STQM01901'  # Life satisfaction
        ]
        if col_name in two_digit_vars:
            return [97, 98, 99]

        # Sub-rule 5c: Default 1-digit for remaining questionnaire items
        return [7, 8, 9]

    # Rule 6: Weights and WLE Scores
    # Source: Page 20
    if col_name == 'WT2019' or col_name.endswith('_WLE_ADJ') or col_name == 'ARS':
        return [9999]

    # Default: No specific rule found
    return []

def check_student_file(file_path):
    print(f"Loading {file_path}...")
    try:
        df, meta = pyreadstat.read_sav(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Checking {len(df.columns)} columns in Student dataset...\n")
    
    found_issues = {}

    for col in df.columns:
        codes = get_student_missing_codes(col)
        if not codes:
            continue
            
        # Check values
        # We drop NaNs first, then check if any remaining values match the codes
        actual_values = df[col].dropna().unique()
        found_codes = [code for code in codes if code in actual_values]
        
        if found_codes:
            found_issues[col] = found_codes

    # --- REPORTING ---
    print("-" * 60)
    print(f"SUMMARY OF ISSUES FOUND (Student Data)")
    print("-" * 60)
    
    if len(found_issues) == 0:
        print("SUCCESS: No raw missing values found. Dataset appears clean.")
    else:
        print(f"WARNING: Found raw missing values in {len(found_issues)} columns.")
        print("Variable Name | Found Codes")
        print("-" * 40)
        for col, codes in found_issues.items():
            print(f"{col:<13} | {codes}")
    
    print("-" * 60)


# Run the check
# Ensure the filename matches your local file
check_student_file(os.path.join(get_data_path(),
                                "INT_01_ST_(2021.04.14)_Public.sav"))
