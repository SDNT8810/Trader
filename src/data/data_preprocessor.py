import pandas as pd
import numpy as np
import talib
from typing import Dict, Any
import os
import yaml
from pathlib import Path

class DataPreprocessor:
    """
    Data preprocessor for Gold price data with technical indicators
    
    Features are configured in config.yaml:
    1. Ichimoku Cloud (H1)
    2. MACD (H1)
    3. Parabolic SAR (H1)
    4. EMA indicators
    5. Bollinger Bands (H1)
    6. RSI indicators
    7. Stochastic indicators
    8. CCI indicator
    9. MFI indicator
    10. ATR indicator
    11. ADX indicators
    12. Momentum indicators
    13. ROC indicators
    14. Williams %R indicator
    """
    
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
        
        if 'indicators' not in self.config['normalization']:
            raise ValueError("Indicator configuration not found in config")
    
    def load_data(self) -> None:
        """Load and prepare the raw data"""
        # Read CSV file
        self.data = pd.read_csv(self.input_file)
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert to float type
        for col in required_columns:
            self.data[col] = self.data[col].astype(float)
    
    def calculate_ichimoku(self) -> None:
        """Calculate Ichimoku Cloud indicators"""
        high = self.data['High']
        low = self.data['Low']
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = talib.MAX(high, timeperiod=9)
        period9_low = talib.MIN(low, timeperiod=9)
        self.data['tenkan_sen'] = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = talib.MAX(high, timeperiod=26)
        period26_low = talib.MIN(low, timeperiod=26)
        self.data['kijun_sen'] = (period26_high + period26_low) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = talib.MAX(high, timeperiod=52)
        period52_low = talib.MIN(low, timeperiod=52)
        self.data['senkou_span_b'] = (period52_high + period52_low) / 2
    
    def calculate_macd(self) -> None:
        """Calculate MACD indicators"""
        close = self.data['Close']
        
        # Calculate MACD, Signal, and Histogram
        macd, signal, hist = talib.MACD(
            close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        self.data['macd'] = macd
        self.data['macd_signal'] = signal
        self.data['macd_hist'] = hist
    
    def calculate_psar(self) -> None:
        """Calculate Parabolic SAR"""
        high = self.data['High']
        low = self.data['Low']
        
        # Default PSAR parameters
        step = 0.02
        maximum = 0.2
        
        # Calculate PSAR
        self.data['psar'] = talib.SAR(
            high,
            low,
            acceleration=step,
            maximum=maximum
        )
    
    def calculate_ema(self) -> None:
        """Calculate EMA indicators"""
        close = self.data['Close']
        
        # Default EMA periods
        ema_periods = [50, 200]
        
        # Calculate EMAs
        for period in ema_periods:
            self.data[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
    
    def calculate_bollinger_bands(self) -> None:
        """Calculate Bollinger Bands"""
        close = self.data['Close']
        
        # Default Bollinger Bands parameters
        period = 20
        nbdev = 2
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=period,
            nbdevup=nbdev,
            nbdevdn=nbdev,
            matype=0
        )
        
        self.data['bb_upper'] = upper
        self.data['bb_middle'] = middle
        self.data['bb_lower'] = lower
    
    def calculate_rsi(self) -> None:
        """Calculate RSI indicators"""
        close = self.data['Close']
        
        # Calculate RSI for different periods
        self.data['rsi_14'] = talib.RSI(close, timeperiod=14)
        self.data['rsi_28'] = talib.RSI(close, timeperiod=28)
    
    def calculate_stochastic(self) -> None:
        """Calculate Stochastic indicators"""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        # Calculate Stochastic
        slowk, slowd = talib.STOCH(
            high,
            low,
            close,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        
        self.data['stoch_k'] = slowk
        self.data['stoch_d'] = slowd
    
    def calculate_cci(self) -> None:
        """Calculate CCI indicator"""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        self.data['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
    
    def calculate_mfi(self) -> None:
        """Calculate Money Flow Index"""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        volume = self.data['Volume']
        
        self.data['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
    
    def calculate_atr(self) -> None:
        """Calculate Average True Range"""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        self.data['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
    
    def calculate_adx(self) -> None:
        """Calculate ADX indicators"""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        # Calculate ADX
        adx = talib.ADX(high, low, close, timeperiod=14)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        self.data['adx_14'] = adx
        self.data['plus_di_14'] = plus_di
        self.data['minus_di_14'] = minus_di
    
    def calculate_momentum(self) -> None:
        """Calculate Momentum indicators"""
        close = self.data['Close']
        
        # Calculate Momentum for different periods
        self.data['mom_10'] = talib.MOM(close, timeperiod=10)
        self.data['mom_20'] = talib.MOM(close, timeperiod=20)
    
    def calculate_roc(self) -> None:
        """Calculate Rate of Change indicators"""
        close = self.data['Close']
        
        # Calculate ROC for different periods
        self.data['roc_10'] = talib.ROC(close, timeperiod=10)
        self.data['roc_20'] = talib.ROC(close, timeperiod=20)
    
    def calculate_williams(self) -> None:
        """Calculate Williams %R indicator"""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        self.data['willr_14'] = talib.WILLR(high, low, close, timeperiod=14)
    
    def process_data(self) -> pd.DataFrame:
        """
        Process the data and calculate all indicators
        
        Returns:
        --------
        pd.DataFrame
            Processed data with all indicators
        """
        # Load data
        self.load_data()
        
        # Calculate indicators
        self.calculate_ichimoku()
        self.calculate_macd()
        self.calculate_psar()
        self.calculate_ema()
        self.calculate_bollinger_bands()
        self.calculate_rsi()
        self.calculate_stochastic()
        self.calculate_cci()
        self.calculate_mfi()
        self.calculate_atr()
        self.calculate_adx()
        self.calculate_momentum()
        self.calculate_roc()
        self.calculate_williams()
        
        # Remove NaN values
        self.data = self.data.dropna()
        
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
        print(f"Processed data saved to {output_file}")

def main():
    """Main function to process the data"""
    # Create preprocessor
    preprocessor = DataPreprocessor()
    
    # Process data
    processed_data = preprocessor.process_data()
    
    # Print data info
    print("\nProcessed Data Info:")
    print(f"Shape: {processed_data.shape}")
    print("\nColumns:")
    for col in processed_data.columns:
        print(f"- {col}")
    
    # Save data
    preprocessor.save_data()

if __name__ == "__main__":
    main()