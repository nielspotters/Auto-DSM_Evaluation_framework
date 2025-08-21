"""
Koh_csv_value_transform
-----------------------
This script reads DSM (Design Structure Matrix) CSV files, applies simple
value transformations, and saves them as Excel (.xlsx) files.

Workflow:
1. Reads all CSV files matching 'dsm_*.csv' inside the 'Data' folder.
2. Standardizes delimiters (',' → ';' if needed).
3. Converts values to integers and applies transformations allowing distribution later on:
   - 0 → -1
   - 5 → 0
4. Saves cleaned DSMs as Excel files in the 'Data_xlsx' folder.
"""

import pandas as pd
import os
import glob
from io import StringIO


def read_and_format_dsm(file_path):
    """
    Read and format a single DSM CSV file.

    - Ensures delimiter is standardized.
    - Converts values to integer type.
    - Applies specific value transformations.

    Args:
        file_path (str): Path to the DSM CSV file.

    Returns:
        pd.DataFrame | None: Processed dataframe or None if error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            first_line = infile.readline()
            infile.seek(0)

            # Detect delimiter: if no ';' found, assume ',' and replace
            if ';' in first_line:
                lines = infile.readlines()
            else:
                lines = [line.replace(',', ';') for line in infile.readlines()]

        # Load into dataframe
        text_data = ''.join(lines)
        df = pd.read_csv(StringIO(text_data), sep=';', index_col=0)

        # Ensure numeric and apply transformations for consistency
        df = df.astype(float).round().astype(int)
        df.replace(0, -1, inplace=True)  # Map 0 → -1
        df.replace(5, 0, inplace=True)   # Map 5 → 0

        return df

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def process_all_dsm_files():
    """
    Process all DSM CSV files in the 'Data' folder and
    save the cleaned results as Excel files in 'Data_xlsx'.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_dir, "Data")
    output_folder = os.path.join(base_dir, "Data_xlsx")

    # Collect all DSM files in the input folder
    dsm_files = glob.glob(os.path.join(data_folder, "dsm_*.csv"))
    print(f"Found {len(dsm_files)} DSM file(s) to process in {data_folder}.")

    for file_path in dsm_files:
        print(f"Processing: {file_path}")
        df = read_and_format_dsm(file_path)

        if df is not None:
            base_filename = os.path.basename(file_path)
            # Output naming: replace prefix and extension
            new_filename = base_filename.replace("dsm_", "Koh_DSM_").replace(".csv", ".xlsx")
            output_path = os.path.join(output_folder, new_filename)

            df.to_excel(output_path, sheet_name="DSM", index=True)
            print(f"Saved cleaned DSM as Excel to: {output_path}")
        else:
            print(f"Skipped file due to error: {file_path}")


if __name__ == "__main__":
    process_all_dsm_files()
