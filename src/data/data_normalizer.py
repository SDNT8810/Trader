import numpy as np
import yaml
from typing import Dict, List, Tuple, Union
import os
import shutil
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

class DataNormalizer:
    """
    Normalize technical indicators using configurable methods
    
    Features:
    - Supports multiple normalization methods
    - Configurable through config.yaml
    - Calculates differentials of price data over index
    - Excludes GMT time from normalization
    - Calculates differentials for specific indicators
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the normalizer
        
        Parameters:
        -----------
        config_path : str
            Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._validate_config()
        self.scalers = {}
        
        # Define indicators that need differential calculation
        self.diff_indicators = {
            'price': ['Open', 'High', 'Low', 'Close'],  # Price data will be differentials
            'ema': ['ema_50', 'ema_200'],
            'bollinger': ['bb_upper', 'bb_middle', 'bb_lower'],
            'psar': ['psar'],
            'ichimoku': ['tenkan_sen', 'kijun_sen', 'senkou_span_b']
        }
    
    def create_fig_directory():
        """Create directory for saving figures if it doesn't exist."""
        fig_dir = 'src/Figs/Normalized'
        if os.path.exists(fig_dir):
            shutil.rmtree(fig_dir)
        os.makedirs(fig_dir)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self) -> None:
        """Validate the configuration structure."""
        if 'normalization' not in self.config:
            raise ValueError("Normalization configuration not found in config file")
        
        if 'methods' not in self.config['normalization']:
            raise ValueError("Normalization methods not found in config")
        
        if 'indicators' not in self.config['normalization']:
            raise ValueError("Indicator configuration not found in config")
    
    def _calculate_differential(self, data: np.ndarray, time_step: float = 1.0) -> np.ndarray:
        """
        Calculate the differential (dx/dt) of a time series
        
        Parameters:
        -----------
        data : np.ndarray
            Time series data
        time_step : float
            Time step between samples
            
        Returns:
        --------
        np.ndarray
            Differential of the time series
        """
        # Calculate first-order difference
        diff = np.diff(data, prepend=data[0])
        # Normalize by time step
        return diff / time_step
    
    def _calculate_price_differences(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate price differences
        
        Parameters:
        -----------
        data : Dict[str, np.ndarray]
            Dictionary of data arrays
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary of price differences
        """
        price_diffs = {}
        price_diffs['high_low'] = data['High'] - data['Low']
        price_diffs['close_open'] = data['Close'] - data['Open']
        price_diffs['high_open'] = data['High'] - data['Open']
        price_diffs['low_open'] = data['Low'] - data['Open']
        return price_diffs
    
    def fit(self, data: Dict[str, np.ndarray]) -> None:
        """
        Fit the normalizer to the data
        
        Parameters:
        -----------
        data : Dict[str, np.ndarray]
            Dictionary of data arrays
        """
        # Get normalization method
        method = list(self.config['normalization']['methods'].keys())[0]
        params = self.config['normalization']['methods'][method]
        
        # Initialize scalers for each indicator group
        for group, indicators in self.config['normalization']['indicators'].items():
            for indicator in indicators:
                if indicator in data:
                    # Check if this indicator needs differential calculation
                    needs_diff = any(indicator in indicators for indicators in self.diff_indicators.values())
                    
                    if needs_diff:
                        # Calculate differential for this indicator
                        diff_data = self._calculate_differential(data[indicator])
                        data_to_fit = diff_data
                    else:
                        data_to_fit = data[indicator]
                    
                    if method == 'min_max':
                        # Convert range list to tuple for MinMaxScaler
                        feature_range = tuple(params['range'])
                        self.scalers[indicator] = MinMaxScaler(feature_range=feature_range)
                    elif method == 'z_score':
                        self.scalers[indicator] = StandardScaler()
                    else:
                        raise ValueError(f"Unknown normalization method: {method}")
                    
                    # Fit the scaler
                    self.scalers[indicator].fit(data_to_fit.reshape(-1, 1))
    
    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform the data using fitted scalers
        
        Parameters:
        -----------
        data : Dict[str, np.ndarray]
            Dictionary of data arrays
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary of normalized data arrays
        """
        normalized_data = {}
        
        # Calculate price differences first
        price_diffs = self._calculate_price_differences(data)
        
        # Handle all indicators including price data
        for group, indicators in self.config['normalization']['indicators'].items():
            for indicator in indicators:
                if indicator in data and indicator in self.scalers:
                    # Check if this indicator needs differential calculation
                    needs_diff = any(indicator in indicators for indicators in self.diff_indicators.values())
                    
                    if needs_diff:
                        # Calculate differential for this indicator
                        diff_data = self._calculate_differential(data[indicator])
                        data_to_transform = diff_data
                    else:
                        data_to_transform = data[indicator]
                    
                    # Normalize the data
                    normalized_data[indicator] = self.scalers[indicator].transform(
                        data_to_transform.reshape(-1, 1)
                    ).flatten()
        
        # Add price differences to normalized data
        normalized_data.update(price_diffs)
        
        return normalized_data
    
    def fit_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Fit and transform the data in one step
        
        Parameters:
        -----------
        data : Dict[str, np.ndarray]
            Dictionary of data arrays
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary of normalized data arrays
        """
        self.fit(data)
        return self.transform(data)
    
    def save_params(self, path: str = "data/normalization_params.npz") -> None:
        """Save normalization parameters to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **self.scalers)
    
    def load_params(self, path: str = "data/normalization_params.npz") -> None:
        """Load normalization parameters from file."""
        if os.path.exists(path):
            params = np.load(path)
            self.scalers = {k: params[k] for k in params.files}
        else:
            raise FileNotFoundError(f"Normalization parameters file not found at {path}") 
        

def main():
    """Main function to normalize the data"""
    # Create normalizer
    normalizer = DataNormalizer()
    
    # Load and prepare data
    print("\nLoading data from Data.csv...")
    data = pd.read_csv('Data.csv')
    
    # Store Gmt time separately
    gmt_time = data['Gmt time']
    
    # Remove NaN rows
    data = data.dropna()
    print(f"Removed NaN rows. New shape: {data.shape}")
    
    # Create data dictionary excluding Gmt time
    data_dict = {col: data[col].values for col in data.columns if col != 'Gmt time'}
    
    # Fit and transform data
    print("Normalizing data...")
    normalizer.fit(data_dict)
    normalized_data = normalizer.transform(data_dict)
    
    # Create DataFrame with normalized data
    normalized_df = pd.DataFrame(normalized_data)
    
    # Add time data back to the DataFrame without normalization
    normalized_df['Gmt time'] = gmt_time
    
    # Save normalized data with time
    normalized_df.to_csv('NData.csv', index=True)
    
    # Print info
    print("\nNormalized Data Info:")
    print(f"Shape: {normalized_df.shape}")
    print("\nColumns:")
    for col in normalized_df.columns:
        print(f"- {col}")
    
    # Print value ranges for a few key indicators
    print("\nValue ranges for key indicators:")
    key_indicators = ['macd', 'rsi_14', 'cci_14', 'Close', 'Volume']
    for indicator in key_indicators:
        if indicator in normalized_df.columns:
            values = normalized_df[indicator]
            print(f"{indicator}: min={values.min():.3f}, max={values.max():.3f}")
    
    print("\nData normalized successfully and saved to NData.csv")
    print("Time data included in NData.csv without normalization")

if __name__ == "__main__":
    main()
