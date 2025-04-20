import pandas as pd
import numpy as np
from typing import Dict, List
import talib
from sklearn.feature_selection import mutual_info_regression

class TechnicalIndicators:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize TechnicalIndicators with OHLCV data.
        
        Args:
            data (pd.DataFrame): DataFrame containing OHLCV data
        """
        self.data = data
        self.open = data['Open'].values
        self.high = data['High'].values
        self.low = data['Low'].values
        self.close = data['Close'].values
        self.volume = data['Volume'].values if 'Volume' in data.columns else None

    def calculate_moving_averages(self) -> pd.DataFrame:
        """Calculate various moving averages."""
        ma_periods = [5, 10, 20, 50, 100, 200]
        indicators = pd.DataFrame()
        
        for period in ma_periods:
            # Simple Moving Average
            indicators[f'SMA_{period}'] = talib.SMA(self.close, timeperiod=period)
            # Exponential Moving Average
            indicators[f'EMA_{period}'] = talib.EMA(self.close, timeperiod=period)
            # Double Exponential Moving Average
            indicators[f'DEMA_{period}'] = talib.DEMA(self.close, timeperiod=period)
            # Triple Exponential Moving Average
            indicators[f'TEMA_{period}'] = talib.TEMA(self.close, timeperiod=period)
            
        return indicators

    def calculate_momentum_indicators(self) -> pd.DataFrame:
        """Calculate momentum indicators."""
        indicators = pd.DataFrame()
        
        # RSI
        indicators['RSI_12'] = talib.RSI(self.close, timeperiod=12)
        indicators['RSI_14'] = talib.RSI(self.close, timeperiod=14)
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators['MACD_12_26_9'] = macd
        indicators['MACD_Signal'] = macdsignal
        indicators['MACD_Hist'] = macdhist
        
        # Stochastic
        slowk, slowd = talib.STOCH(self.high, self.low, self.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        indicators['STOCH_K'] = slowk
        indicators['STOCH_D'] = slowd
        
        # CCI
        indicators['CCI_14'] = talib.CCI(self.high, self.low, self.close, timeperiod=14)
        
        # MFI (if volume data is available)
        if self.volume is not None:
            indicators['MFI_14'] = talib.MFI(self.high, self.low, self.close, self.volume, timeperiod=14)
        
        return indicators

    def calculate_volatility_indicators(self) -> pd.DataFrame:
        """Calculate volatility indicators."""
        indicators = pd.DataFrame()
        
        # Bollinger Bands
        upperband, middleband, lowerband = talib.BBANDS(self.close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        indicators['BB_upper_20_2'] = upperband
        indicators['BB_middle_20_2'] = middleband
        indicators['BB_lower_20_2'] = lowerband
        
        # ATR
        indicators['ATR_14'] = talib.ATR(self.high, self.low, self.close, timeperiod=14)
        
        return indicators

    def calculate_trend_indicators(self) -> pd.DataFrame:
        """Calculate trend indicators."""
        indicators = pd.DataFrame()
        
        # ADX
        indicators['ADX_14'] = talib.ADX(self.high, self.low, self.close, timeperiod=14)
        indicators['PLUS_DI_14'] = talib.PLUS_DI(self.high, self.low, self.close, timeperiod=14)
        indicators['MINUS_DI_14'] = talib.MINUS_DI(self.high, self.low, self.close, timeperiod=14)
        
        return indicators

    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators."""
        all_indicators = pd.DataFrame()
        
        # Calculate each group of indicators
        moving_averages = self.calculate_moving_averages()
        momentum = self.calculate_momentum_indicators()
        volatility = self.calculate_volatility_indicators()
        trend = self.calculate_trend_indicators()
        
        # Combine all indicators
        all_indicators = pd.concat([
            moving_averages,
            momentum,
            volatility,
            trend
        ], axis=1)
        
        return all_indicators

    def get_feature_importance(self, target: pd.Series) -> pd.Series:
        """
        Calculate feature importance using mutual information.
        
        Args:
            target (pd.Series): Target variable for importance calculation
            
        Returns:
            pd.Series: Feature importance scores
        """
        # Get all indicators
        features = self.calculate_all_indicators()
        
        # Remove any remaining NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')
        target = target.fillna(method='ffill').fillna(method='bfill')
        
        # Align features and target
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]
        
        # Calculate mutual information scores
        mi_scores = mutual_info_regression(features, target)
        
        # Create a series with feature names and scores
        feature_importance = pd.Series(mi_scores, index=features.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        
        return feature_importance 