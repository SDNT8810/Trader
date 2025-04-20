import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataNormalizer:
    def __init__(self, input_file='Data.csv', output_file='NData.csv'):
        self.input_file = input_file
        self.output_file = output_file
        self.data = pd.read_csv(self.input_file)
        self.normalized = pd.DataFrame()

    def normalize(self):
        # Clean data
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data.reset_index(drop=True, inplace=True)
        self.normalized["Index"] = self.data.index

        # Calculate price range for normalization
        price_range = self.data['High'] - self.data['Low']
        avg_price = (self.data['High'] + self.data['Low']) / 2

        for col in self.data.columns:
            if col == "Index" or col.lower() in ["time", "date"]:
                continue

            series = self.data[col]
            
            # Skip columns that are all NaN
            if series.isna().all():
                self.normalized[col] = 0
                continue

            # Handle different types of indicators
            if col.startswith(("RSI", "MFI")):
                # RSI and MFI are naturally bounded between 0-100
                self.normalized[col] = series / 100.0
            elif col.startswith("WILLR"):
                # WILLR is naturally bounded between -100 to 0
                self.normalized[col] = (series + 100) / 100.0
            elif col.startswith(("CCI", "MACD", "MOM", "ROC")):
                # These oscillators are unbounded but typically range between -100 to 100
                max_abs = max(abs(series.min()), abs(series.max()))
                self.normalized[col] = series / max_abs
            elif col.startswith(("ADX", "DI")):
                # ADX and DI are naturally bounded between 0-100
                self.normalized[col] = series / 100.0
            elif col.startswith("BB_"):
                # Bollinger Bands: normalize relative to price range
                if col.endswith("_upper"):
                    self.normalized[col] = (series - self.data['Close']) / price_range
                elif col.endswith("_lower"):
                    self.normalized[col] = (series - self.data['Close']) / price_range
                else:  # BB_middle
                    self.normalized[col] = (series - self.data['Close']) / price_range
            elif col.startswith(("ATR", "NATR")):
                # ATR and NATR normalized relative to price range
                self.normalized[col] = series / price_range
            elif col.startswith("Volume"):
                # Volume normalized relative to its own moving average
                ma_volume = series.rolling(window=20, min_periods=1).mean()
                self.normalized[col] = series / ma_volume
            elif any(x in col for x in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA']):
                # Moving averages: show relative position to price
                self.normalized[col] = (series - self.data['Close']) / price_range
            elif col in ['Open', 'High', 'Low', 'Close']:
                # Price indicators: normalize relative to average price
                self.normalized[col] = (series - avg_price) / price_range
            else:
                # Default to -1 to 1 range for other indicators
                scaler = MinMaxScaler(feature_range=(-1, 1))
                valid_data = series.dropna()
                if not valid_data.empty:
                    scaled = scaler.fit_transform(valid_data.values.reshape(-1, 1)).flatten()
                    self.normalized[col] = pd.Series(scaled, index=valid_data.index)
                else:
                    self.normalized[col] = 0

            # Fill any remaining NaN values with 0
            self.normalized[col] = self.normalized[col].fillna(0)

    def save(self):
        self.normalized.to_csv(self.output_file, index=False)

    def run(self):
        self.normalize()
        self.save()

if __name__ == "__main__":
    normalizer = DataNormalizer()
    normalizer.run()