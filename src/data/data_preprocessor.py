import pandas as pd
import numpy as np
import talib
from typing import Dict, Any
import os
import yaml
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    
    def __init__(self, input_file: str = 'Gchart.csv', config_path: str = 'config/config.yaml'):
        """
        Initialize the preprocessor
        
        Parameters:
        -----------
        input_file : str
            Path to the input CSV file containing gold price data
        config_path : str
            Path to the configuration file
        """
        self.input_file = input_file
        self.data = None
        self.config = self._load_config(config_path)
        self._validate_config()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self) -> None:
        """Validate the configuration structure."""
        if 'normalization' not in self.config:
            raise ValueError("Normalization configuration not found in config file")
        
        if not all(key in self.config['normalization'] for key in ['price_features', 'ratio_features', 'indicator_features']):
            raise ValueError("Feature group configurations not found in config")
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        print("Calculating technical indicators...")
        
        # 1. Ichimoku Cloud
        print("- Ichimoku Cloud")
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        # Tenkan-sen (Conversion Line)
        period9_high = pd.Series(high).rolling(window=9).max()
        period9_low = pd.Series(low).rolling(window=9).min()
        data['Tenkan_sen'] = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line)
        period26_high = pd.Series(high).rolling(window=26).max()
        period26_low = pd.Series(low).rolling(window=26).min()
        data['Kijun_sen'] = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A)
        data['Senkou_Span_A'] = ((data['Tenkan_sen'] + data['Kijun_sen']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        period52_high = pd.Series(high).rolling(window=52).max()
        period52_low = pd.Series(low).rolling(window=52).min()
        data['Senkou_Span_B'] = ((period52_high + period52_low) / 2).shift(26)
        
        # 2. MACD
        print("- MACD")
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # 3. Parabolic SAR
        print("- PSAR")
        data['PSAR'] = talib.SAR(data['High'], data['Low'], 
                                acceleration=0.02, maximum=0.2)
        
        # 4. EMA 50
        print("- EMA 50")
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        
        # 5. EMA 200
        print("- EMA 200")
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()
        
        # 6. Bollinger Bands
        print("- Bollinger Bands")
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
        
        # Calculate additional features
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
        data['EMA_50_200_Ratio'] = data['EMA_50'] / data['EMA_200']
        data['Price_EMA_50_Ratio'] = data['Close'] / data['EMA_50']
        data['Price_EMA_200_Ratio'] = data['Close'] / data['EMA_200']
        
        # Fill NaN values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        print("Technical indicators calculation completed.")
        return data
    
    def normalize_features(self, data):
        """Normalize features using MinMaxScaler"""
        print("\nNormalizing features...")
        
        # Initialize scalers dictionary
        self.scalers = {}
        
        # Get feature groups from config
        price_features = self.config['normalization']['price_features']
        ratio_features = self.config['normalization']['ratio_features']
        indicator_features = self.config['normalization']['indicator_features']
        
        # Create normalized data DataFrame
        normalized_data = pd.DataFrame(index=data.index)
        
        # Normalize price features (0 to 1 range)
        for feature in price_features:
            self.scalers[feature] = MinMaxScaler()
            normalized_data[feature] = self.scalers[feature].fit_transform(data[[feature]]).flatten()
        
        # Normalize ratio features (0 to 2 range)
        for feature in ratio_features:
            self.scalers[feature] = MinMaxScaler(feature_range=(0, 2))
            normalized_data[feature] = self.scalers[feature].fit_transform(data[[feature]]).flatten()
        
        # Normalize indicator features (-1 to 1 range)
        for feature in indicator_features:
            self.scalers[feature] = MinMaxScaler(feature_range=(-1, 1))
            normalized_data[feature] = self.scalers[feature].fit_transform(data[[feature]]).flatten()
        
        # Add target (Mean_Price) with standard normalization
        self.scalers['Mean_Price'] = MinMaxScaler()
        normalized_data['Mean_Price'] = self.scalers['Mean_Price'].fit_transform(data[['Mean_Price']]).flatten()
        
        print(f"\nFinal processed data shape: {normalized_data.shape}")
        print(f"Features ({len(normalized_data.columns)-1}):")
        for col in normalized_data.columns:
            if col != 'Mean_Price':
                print(f"- {col}")
        
        return normalized_data
    
    def load_data(self):
        """Load and preprocess the data"""
        try:
            print("\nLoading and preprocessing data...")
            # Load data from input file
            data = pd.read_csv(self.input_file)
            print(f"Raw data shape: {data.shape}")
            
            # Select only HOCL and Volume columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = data[required_columns]
            
            # Calculate mean price (average of HOCL)
            data['Mean_Price'] = data[['High', 'Open', 'Close', 'Low']].mean(axis=1)
            
            # Calculate technical indicators
            data = self.calculate_technical_indicators(data)
            
            # Store original data for later use
            self.original_data = data.copy()
            
            # Normalize features
            self.data = self.normalize_features(data)
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_sequences(self, data):
        """Create input sequences and targets for trading signal prediction"""
        X, y = [], []
        window_size = self.config['model']['window_size']
        prediction_window = self.config['model']['prediction_window']
        
        # Get all feature columns except Mean_Price
        feature_columns = [col for col in data.columns if col != 'Mean_Price']
        print(f"\nCreating sequences using {len(feature_columns)} features:")
        for col in feature_columns:
            print(f"- {col}")
        
        for i in range(len(data) - window_size - prediction_window + 1):
            # Input sequence: All features for past window_size time steps
            X.append(data.iloc[i:i+window_size][feature_columns].values)
            
            # Get current price and future prices
            current_price = data.iloc[i+window_size-1]['Mean_Price']
            future_prices = data.iloc[i+window_size:i+window_size+prediction_window]['Mean_Price'].values
            
            # Calculate various statistics for the prediction window
            mean_future = np.mean(future_prices)
            max_future = np.max(future_prices)
            min_future = np.min(future_prices)
            sum_future = np.sum(future_prices)
            
            # Calculate potential returns for different strategies
            mean_return = (mean_future - current_price) / current_price
            max_return = (max_future - current_price) / current_price
            min_return = (min_future - current_price) / current_price
            sum_return = (sum_future - current_price * prediction_window) / (current_price * prediction_window)
            
            # Determine the best trading signal based on all statistics
            # We'll use a weighted combination of all returns
            weights = np.array([0.4, 0.3, 0.2, 0.1])  # Weights for mean, max, min, sum returns
            returns = np.array([mean_return, max_return, min_return, sum_return])
            weighted_return = np.sum(weights * returns)
            
            # Create trading signal based on weighted return
            # Scale the return to make signals more pronounced
            signal = np.clip(weighted_return * 5, -1, 1)
            
            # If the signal is very close to 0, set it to 0 (do nothing)
            if abs(signal) < 0.1:
                signal = 0
            
            y.append(signal)
        
        X = np.array(X)
        y = np.array(y)
        print(f"\nSequence shapes:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        return X, y
    
    def process_data(self) -> pd.DataFrame:
        """
        Process the data and calculate all indicators
        
        Returns:
        --------
        pd.DataFrame
            Processed data with all indicators
        """
        # Load and preprocess data
        self.data = self.load_data()
        
        # Remove NaN values
        if self.data is not None:
            self.data = self.data.dropna()
            print(f"\nFinal data shape after removing NaN: {self.data.shape}")
        
        return self.data
    
    def save_data(self, output_file: str = 'Data.csv') -> None:
        """
        Save processed data to CSV file
        
        Parameters:
        -----------
        output_file : str
            Path to save the processed data
        """
        if self.data is None:
            raise ValueError("No data to save. Run process_data() first.")
        
        self.data.to_csv(output_file, index=False)
        print(f"\nProcessed data saved to {output_file}")

def main():
    """Main function to process the data"""
    # Create preprocessor
    preprocessor = DataPreprocessor()
    
    # Process data
    processed_data = preprocessor.process_data()
    
    if processed_data is not None:
        # Save data
        preprocessor.save_data()
    else:
        print("Error: Data processing failed.")

if __name__ == "__main__":
    main()