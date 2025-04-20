
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
        self.data = self.data.replace([np.inf, -np.inf], np.nan).dropna()
        self.data.reset_index(drop=True, inplace=True)
        self.normalized["Index"] = self.data.index

        for col in self.data.columns:
            if col == "Index" or col.lower() in ["time", "date"]:
                continue

            series = self.data[col]
            finite_series = series.replace([np.inf, -np.inf], np.nan).dropna()

            if finite_series.empty:
                self.normalized[col] = 0
                continue

            # Decide normalization range
            if col.startswith(("RSI", "MFI", "WILLR", "ULTOSC")):
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                scaler = MinMaxScaler(feature_range=(-1, 1))

            scaled = scaler.fit_transform(series.fillna(series.median()).values.reshape(-1, 1)).flatten()
            self.normalized[col] = scaled

    def save(self):
        self.normalized.to_csv(self.output_file, index=False)

    def run(self):
        self.normalize()
        self.save()

if __name__ == "__main__":
    normalizer = DataNormalizer()
    normalizer.run()
