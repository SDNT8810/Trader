import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional

class TechnicalIndicators:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize TechnicalIndicators with price data.
        
        Args:
            data (pd.DataFrame): DataFrame containing OHLC data
        """
        self.data = data
        self.indicators: Dict[str, pd.Series] = {}
        
    def calculate_moving_averages(self) -> Dict[str, pd.Series]:
        """Calculate various moving averages."""
        ma_periods = [5, 10, 20]
        
        for period in ma_periods:
            # Simple Moving Average
            self.indicators[f'SMA_{period}'] = talib.SMA(self.data['Close'], timeperiod=period)
            
            # Exponential Moving Average
            self.indicators[f'EMA_{period}'] = talib.EMA(self.data['Close'], timeperiod=period)
            
            # Weighted Moving Average
            self.indicators[f'WMA_{period}'] = talib.WMA(self.data['Close'], timeperiod=period)
            
        return {k: v for k, v in self.indicators.items() if k.startswith(('SMA', 'EMA', 'WMA'))}
        
    def calculate_momentum_indicators(self) -> Dict[str, pd.Series]:
        """Calculate momentum indicators."""
        # Relative Strength Index
        self.indicators['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            self.data['Close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        self.indicators['MACD'] = macd
        self.indicators['MACD_Signal'] = macd_signal
        self.indicators['MACD_Hist'] = macd_hist
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(
            self.data['High'],
            self.data['Low'],
            self.data['Close'],
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        self.indicators['Stoch_K'] = slowk
        self.indicators['Stoch_D'] = slowd
        
        # Williams %R
        self.indicators['Williams_R'] = talib.WILLR(
            self.data['High'],
            self.data['Low'],
            self.data['Close'],
            timeperiod=14
        )
        
        return {k: v for k, v in self.indicators.items() if k in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D', 'Williams_R']}
        
    def calculate_volatility_indicators(self) -> Dict[str, pd.Series]:
        """Calculate volatility indicators."""
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            self.data['Close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        self.indicators['BB_Upper'] = upper
        self.indicators['BB_Middle'] = middle
        self.indicators['BB_Lower'] = lower
        
        # Average True Range
        self.indicators['ATR'] = talib.ATR(
            self.data['High'],
            self.data['Low'],
            self.data['Close'],
            timeperiod=14
        )
        
        # Keltner Channels
        atr = self.indicators['ATR']
        ema = talib.EMA(self.data['Close'], timeperiod=20)
        self.indicators['KC_Upper'] = ema + (2 * atr)
        self.indicators['KC_Middle'] = ema
        self.indicators['KC_Lower'] = ema - (2 * atr)
        
        return {k: v for k, v in self.indicators.items() if k.startswith(('BB_', 'ATR', 'KC_'))}
        
    def calculate_trend_indicators(self) -> Dict[str, pd.Series]:
        """Calculate trend indicators."""
        # Average Directional Index
        self.indicators['ADX'] = talib.ADX(
            self.data['High'],
            self.data['Low'],
            self.data['Close'],
            timeperiod=14
        )
        
        # Parabolic SAR
        self.indicators['SAR'] = talib.SAR(
            self.data['High'],
            self.data['Low'],
            acceleration=0.02,
            maximum=0.2
        )
        
        # Ichimoku Cloud
        high_9 = self.data['High'].rolling(window=9).max()
        low_9 = self.data['Low'].rolling(window=9).min()
        self.indicators['Tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = self.data['High'].rolling(window=26).max()
        low_26 = self.data['Low'].rolling(window=26).min()
        self.indicators['Kijun_sen'] = (high_26 + low_26) / 2
        
        self.indicators['Senkou_Span_A'] = ((self.indicators['Tenkan_sen'] + self.indicators['Kijun_sen']) / 2).shift(26)
        
        high_52 = self.data['High'].rolling(window=52).max()
        low_52 = self.data['Low'].rolling(window=52).min()
        self.indicators['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
        
        return {k: v for k, v in self.indicators.items() if k in ['ADX', 'SAR', 'Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B']}
        
    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators and return as DataFrame."""
        self.calculate_moving_averages()
        self.calculate_momentum_indicators()
        self.calculate_volatility_indicators()
        self.calculate_trend_indicators()
        
        # Create DataFrame with all indicators
        indicators_df = pd.DataFrame(self.indicators)
        
        # Handle missing values
        indicators_df = indicators_df.fillna(method='ffill').fillna(method='bfill')
        
        return indicators_df
        
    def get_feature_importance(self, target: pd.Series) -> pd.DataFrame:
        """
        Calculate feature importance using correlation with target.
        
        Args:
            target (pd.Series): Target variable (e.g., next day returns)
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        indicators_df = self.calculate_all_indicators()
        correlations = indicators_df.corrwith(target)
        importance = pd.DataFrame({
            'Feature': correlations.index,
            'Correlation': correlations.values,
            'Absolute_Correlation': abs(correlations.values)
        })
        return importance.sort_values('Absolute_Correlation', ascending=False) 