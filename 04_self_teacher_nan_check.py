import pandas as pd
import numpy as np
import pyreadstat
import os
from Utils import get_data_path


def values_to_replace(desc: str) -> bool:
    missing_strings_to_zero = [
        "N/A",
        "Missing by design",
        "Omitted",
        "Invalid",
        "Not Applicable",
        "Missing session length (missing end date)"
    ]
    return desc in missing_strings_to_zero


def find_null_data(df: pd.DataFrame, meta):
    # print(meta.variable_value_labels)
    fieldID_desc = {}
    for column in df.columns:
        if column not in meta.variable_value_labels.keys():
            continue

        replace_counter = 0
        codes = {}
        for key, value in meta.variable_value_labels[str(column)].items():
            if not values_to_replace(value):
                # print("=" * 15)
                # print(df[column].max)
                # desc = list(meta.variable_value_labels[str(column)].items())[-3:]
                # print(desc)
                # print("=" * 15 + "\n")
                continue
            codes[key] = value
            missing_values_mask = df[column].astype(float) == key
            missing_rows = df.loc[missing_values_mask, column]
            df.loc[missing_values_mask, column] = np.nan
            if missing_rows.count() == 0:
                continue

            replace_counter += 1
            if key in fieldID_desc:
                fieldID_desc[column].append(key)
            else:
                fieldID_desc[column] = [key]

        # print(codes)

    return fieldID_desc


def check_teacher_file(file_path):
    print(f"Loading {file_path}...")
    try:
        df, meta = pyreadstat.read_sav(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Checking {len(df.columns)} columns in Teacher dataset...\n")

    found_issues = find_null_data(df, meta)

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
# check_teacher_file('C:\\Users\\ppp\\Documents\\Witek\\Studia\\Laby\\Research Project\\main project\\data\\INT_02_PA_(2021.04.14)_Public.sav')
check_teacher_file(os.path.join(get_data_path(), 'INT_04_TCQ_(2021.04.14)_Public.sav'))
check_teacher_file(os.path.join(get_data_path(), 'INT_05_PRQ_(2021.04.14)_Public.sav'))