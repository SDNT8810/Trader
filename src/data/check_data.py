import pandas as pd
import numpy as np

def check_data_quality():
    # Read the normalized data
    df = pd.read_csv('NData.csv')
    
    # Print dataset size
    print(f"Dataset Size:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Total cells: {len(df) * len(df.columns)}")
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    print(f"\nNaN Values:")
    print(f"Total NaN values: {nan_count}")
    if nan_count > 0:
        print("\nNaN values per column:")
        nan_cols = df.isna().sum()
        print(nan_cols[nan_cols > 0])
    
    # Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"\nInfinite Values:")
    print(f"Total infinite values: {inf_count}")
    if inf_count > 0:
        print("\nInfinite values per column:")
        inf_cols = np.isinf(df.select_dtypes(include=[np.number])).sum()
        print(inf_cols[inf_cols > 0])
    
    # Check ranges of normalized columns
    print("\nNormalized Column Ranges:")
    out_of_bounds_cols = []
    for col in df.columns:
        if col.endswith('_norm'):
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{col}:")
            print(f"  Min: {min_val:.4f}")
            print(f"  Max: {max_val:.4f}")
            
            # Check if values are within expected ranges
            if any(x in col for x in ['CCI', 'MACD', 'MOM', 'ROC', 'Price_Change', 'APO', 'PPO', 'TRIX']):
                if min_val < -1.0 or max_val > 1.0:
                    print(f"  WARNING: Values outside expected range [-1, 1]")
                    out_of_bounds_cols.append((col, min_val, max_val))
            elif 'WILLR' in col:
                if min_val < -1.0 or max_val > 0.0:
                    print(f"  WARNING: Values outside expected range [-1, 0]")
                    out_of_bounds_cols.append((col, min_val, max_val))
            else:
                if min_val < 0.0 or max_val > 1.0:
                    print(f"  WARNING: Values outside expected range [0, 1]")
                    out_of_bounds_cols.append((col, min_val, max_val))
    
    # Print summary of out-of-bounds columns
    if out_of_bounds_cols:
        print("\nSummary of Out-of-Bounds Columns:")
        for col, min_val, max_val in out_of_bounds_cols:
            print(f"{col}: [{min_val:.4f}, {max_val:.4f}]")
    
    # Check data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows:")
    print(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0:
        print("\nDuplicate rows details:")
        dup_rows = df[df.duplicated(keep=False)]
        print(dup_rows.head())
    
    # Check for missing values in each column
    print("\nMissing Values per Column:")
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if len(missing_cols) > 0:
        print(missing_cols)
    else:
        print("No missing values found in any column")

if __name__ == "__main__":
    check_data_quality() 