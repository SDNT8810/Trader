import pandas as pd
import numpy as np
import os
from tabulate import tabulate
from typing import Dict, List, Tuple
from datetime import datetime

class DataChecker:
    """
    Analyze CSV files in the root directory and report their statistics.
    
    Features:
    - Reports number of rows and columns
    - Counts NaN and Inf values
    - Shows value ranges for each column
    - Analyzes timestamp alignment
    - Shows data overlap information
    """
    
    def __init__(self, root_dir: str = '.'):
        self.root_dir = root_dir
        self.csv_files = self._find_csv_files()
        self.timestamp_col = 'Gmt time'
    
    def _find_csv_files(self) -> List[str]:
        """Find all CSV files in the root directory"""
        return [f for f in os.listdir(self.root_dir) if f.endswith('.csv')]
    
    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """Parse timestamps with European date format"""
        if self.timestamp_col in df.columns:
            return pd.to_datetime(df[self.timestamp_col], format='%d.%m.%Y %H:%M:%S.%f')
        return None
    
    def _analyze_file(self, file_path: str) -> Dict:
        """Analyze a single CSV file and return its statistics"""
        try:
            df = pd.read_csv(file_path)
            stats = {
                'File': file_path,
                'Rows': len(df),
                'Columns': len(df.columns),
                'NaN_Count': df.isna().sum().sum(),
                'Inf_Count': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
                'Value_Ranges': self._get_value_ranges(df)
            }
            
            # Add timestamp analysis if available
            timestamps = self._parse_timestamps(df)
            if timestamps is not None:
                stats['Time_Range'] = f"{timestamps.min()} to {timestamps.max()}"
                stats['Time_Step'] = (timestamps.max() - timestamps.min()) / len(timestamps)
            
            return stats
        except Exception as e:
            return {
                'File': file_path,
                'Error': str(e)
            }
    
    def _get_value_ranges(self, df: pd.DataFrame) -> str:
        """Get value ranges for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return "No numeric columns"
        
        ranges = []
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            ranges.append(f"{col}: [{min_val:.2f}, {max_val:.2f}]")
        
        return "\n".join(ranges)
    
    def _analyze_data_alignment(self) -> Dict:
        """Analyze alignment between different data files"""
        alignment = {}
        
        # Load all files with timestamps
        data_files = {}
        for file in self.csv_files:
            try:
                df = pd.read_csv(file)
                timestamps = self._parse_timestamps(df)
                if timestamps is not None:
                    data_files[file] = timestamps
            except:
                continue
        
        # Compare timestamps between files
        for file1, ts1 in data_files.items():
            for file2, ts2 in data_files.items():
                if file1 != file2:
                    common_ts = set(ts1) & set(ts2)
                    alignment[f"{file1} vs {file2}"] = {
                        'Common_Timestamps': len(common_ts),
                        'File1_Only': len(ts1) - len(common_ts),
                        'File2_Only': len(ts2) - len(common_ts)
                    }
        
        return alignment
    
    def generate_report(self) -> None:
        """Generate and print the analysis report"""
        if not self.csv_files:
            print("No CSV files found in the root directory.")
            return
        
        # Analyze all files
        results = [self._analyze_file(f) for f in self.csv_files]
        
        # Prepare table data
        table_data = []
        for result in results:
            if 'Error' in result:
                table_data.append([
                    result['File'],
                    'Error',
                    result['Error'],
                    '-',
                    '-',
                    '-',
                    '-'
                ])
            else:
                table_data.append([
                    result['File'],
                    f"{result['Rows']} x {result['Columns']}",
                    result['NaN_Count'],
                    result['Inf_Count'],
                    result.get('Time_Range', 'No timestamps'),
                    result.get('Time_Step', 'N/A'),
                    result['Value_Ranges']
                ])
        
        # Print the report
        print("\nCSV Files Analysis Report")
        print("=" * 100)
        print(tabulate(
            table_data,
            headers=['File', 'Shape', 'NaN Count', 'Inf Count', 'Time Range', 'Time Step', 'Value Ranges'],
            tablefmt='grid',
            stralign='left'
        ))
        print("=" * 100)
        
        # Print data alignment analysis
        print("\nData Alignment Analysis")
        print("=" * 100)
        alignment = self._analyze_data_alignment()
        alignment_data = []
        for comparison, stats in alignment.items():
            alignment_data.append([
                comparison,
                stats['Common_Timestamps'],
                stats['File1_Only'],
                stats['File2_Only']
            ])
        
        print(tabulate(
            alignment_data,
            headers=['Comparison', 'Common Timestamps', 'File1 Only', 'File2 Only'],
            tablefmt='grid',
            stralign='left'
        ))
        print("=" * 100)

def main():
    """Run the data checker"""
    checker = DataChecker()
    checker.generate_report()

if __name__ == "__main__":
    main() 