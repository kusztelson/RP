import pandas as pd
import pyreadstat

# Configuration
INPUT_FILE = 'INT_Final_Merged_Prefixed.sav'
OUTPUT_DATA_CSV = 'INT_Final_Merged_Prefixed.csv'
OUTPUT_DICT_CSV = 'INT_Final_Merged_Dictionary.csv'

def export_data_and_dictionary():
    print(f"Loading {INPUT_FILE}...")
    try:
        # 1. Load the SPSS file (getting both Data and Metadata)
        df, meta = pyreadstat.read_sav(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Make sure you ran the previous merge script.")
        return

    # ---------------------------------------------------------
    # PART A: SAVE DATA TO CSV
    # ---------------------------------------------------------
    print(f"Exporting data to {OUTPUT_DATA_CSV}...")
    # index=False removes the row numbers
    df.to_csv(OUTPUT_DATA_CSV, index=False)
    
    # ---------------------------------------------------------
    # PART B: SAVE METADATA TO CSV (The "Data Dictionary")
    # ---------------------------------------------------------
    print(f"Generating Data Dictionary to {OUTPUT_DICT_CSV}...")
    
    # meta.column_names_to_labels is a dictionary: {'ST_VarName': 'Description'}
    # We convert it to a DataFrame for easy saving
    df_dictionary = pd.DataFrame.from_dict(
        meta.column_names_to_labels, 
        orient='index', 
        columns=['Description']
    )
    
    # Reset index so "Variable Name" is a proper column, not just the index
    df_dictionary.index.name = 'Variable_Name'
    df_dictionary.reset_index(inplace=True)
    
    # Save to CSV
    df_dictionary.to_csv(OUTPUT_DICT_CSV, index=False)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("-" * 50)
    print("SUCCESS")
    print("-" * 50)
    print(f"1. Data File:       {OUTPUT_DATA_CSV} ({df.shape[0]} rows)")
    print(f"2. Data Dictionary: {OUTPUT_DICT_CSV} ({df_dictionary.shape[0]} variables)")
    print("-" * 50)
    print("Example Dictionary Entry:")
    print(df_dictionary.head(3).to_string(index=False))

if __name__ == "__main__":
    export_data_and_dictionary()