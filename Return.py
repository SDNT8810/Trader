import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- Settings ---
window_size = 60
prediction_window = 10

# --- Compute returns and labels ---
close_prices = df['Close'].values
X_data = []
y_data = []
returns = []

for i in range(window_size, len(close_prices) - prediction_window):
    window = close_prices[i - window_size:i]
    future_prices = close_prices[i:i + prediction_window]
    future_mean = np.mean(future_prices)
    current_price = close_prices[i]

    X_data.append(df.iloc[i - window_size:i][['High', 'Open', 'Close', 'Low', ...]].values)  # Add your indicator columns
    profit = future_mean - current_price

    # Label logic: based on threshold (you can tune)
    if profit > 0.3:
        label = 2  # Buy
    elif profit < -0.3:
        label = 0  # Sell
    else:
        label = 1  # Hold

    y_data.append(label)
    returns.append(profit)

# --- Convert to Torch ---
X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32)
y_tensor = torch.tensor(np.array(y_data), dtype=torch.long)
returns = np.array(returns)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
