import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from src.utils.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, file_path: str):
        """
        Initialize the DataLoader with the path to the data file.
        
        Args:
            file_path (str): Path to the CSV file containing OHLC data
        """
        self.file_path = file_path
        self.data = None
        self.sequence_length = 10  # Default sequence length for daily trading
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the OHLC data from the CSV file.
        
        Returns:
            pd.DataFrame: DataFrame containing the OHLC data
        """
        try:
            self.data = pd.read_csv(self.file_path)
            # Rename 'Gmt time' to 'Date' and set it as index
            self.data = self.data.rename(columns={'Gmt time': 'Date'})
            self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d.%m.%Y %H:%M:%S.%f')
            logger.info(f"Successfully loaded data from {self.file_path}")
            logger.info(f"Data shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def validate_data(self) -> bool:
        """
        Validate the loaded data for required columns and data types.
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
            
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in self.data.columns for col in required_columns):
            logger.error(f"Missing required columns. Expected: {required_columns}")
            return False
            
        # Check for missing values
        if self.data[required_columns].isnull().any().any():
            logger.warning("Data contains missing values")
            
        return True
        
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the data for daily trading analysis.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        if not self.validate_data():
            raise ValueError("Data validation failed")
            
        # Create daily returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        
        # Create daily volatility (using daily range)
        self.data['Daily_Volatility'] = (self.data['High'] - self.data['Low']) / self.data['Open']
        
        # Calculate technical indicators
        logger.info("Calculating technical indicators...")
        tech_indicators = TechnicalIndicators(self.data)
        indicators_df = tech_indicators.calculate_all_indicators()
        
        # Combine original data with technical indicators
        self.data = pd.concat([self.data, indicators_df], axis=1)
        
        # Handle missing values
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate feature importance
        logger.info("Calculating feature importance...")
        feature_importance = tech_indicators.get_feature_importance(self.data['Daily_Return'].shift(-1))
        logger.info("\nFeature Importance:")
        logger.info(feature_importance.head(10))
        
        logger.info("Data preprocessing completed")
        return self.data
        
    def create_sequences(self, data: pd.DataFrame, sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for the GRU model.
        
        Args:
            data (pd.DataFrame): Preprocessed data
            sequence_length (int, optional): Length of sequences. Defaults to self.sequence_length.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features) and y (targets) arrays
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        # Select features for the model (excluding date and target columns)
        feature_columns = [col for col in data.columns if col not in ['Date', 'Daily_Return']]
        
        # Calculate target variable (future returns)
        future_returns = data['Daily_Return'].shift(-1)
        
        # Create a more balanced target variable
        volatility = data['Daily_Volatility'].rolling(window=20).std().fillna(0)
        threshold = volatility.mean()
        
        y = np.where(future_returns > threshold, 1,
                    np.where(future_returns < -threshold, -1, 0))
        
        # Prepare feature data
        X_data = data[feature_columns].values
        
        X, y_out = [], []
        for i in range(len(data) - sequence_length):
            if i + sequence_length < len(y):  # Check if we have enough future data
                X.append(X_data[i:(i + sequence_length)])
                y_out.append(y[i + sequence_length])
            
        return np.array(X), np.array(y_out)
        
    def get_train_test_split(self, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        X, y = self.create_sequences(self.data)
        
        # Split the data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Testing set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test 