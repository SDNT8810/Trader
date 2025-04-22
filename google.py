import pandas as pd
import numpy as np
import talib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import time # Import time to measure execution

# --- Hyperparameters ---
window_size = 100
prediction_window = 10
epochs = 100
batch_size = 128
learning_rate = 1e-3
# Threshold for buy/sell signal
buy_threshold = 0.002
sell_threshold = -0.002

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load and preprocess ---
print("Loading and preprocessing data...")
start_time = time.time()
df = pd.read_csv('Gchart.csv')
df = df[['High', 'Open', 'Close', 'Low', 'Volume']].copy()

# --- Indicators ---
# Perform TA-Lib calculations
df['EMA50'] = talib.EMA(df['Close'], timeperiod=50)
df['EMA200'] = talib.EMA(df['Close'], timeperiod=200)
macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['MACD'] = macd - macdsignal
df['PSAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
df['BB_upper'] = upper
df['BB_middle'] = middle
df['BB_lower'] = lower
# Use pandas rolling for Ichimoku components
df['Tenkan'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
df['Kijun'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
df['SenkouB'] = (df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Preprocessing complete. Data shape: {df.shape}")
print(f"Preprocessing time: {time.time() - start_time:.4f} seconds")

# --- Feature Engineering ---
print("Feature engineering and scaling...")
start_time = time.time()
features = ['High', 'Open', 'Close', 'Low', 'Volume', 'EMA50', 'EMA200', 'MACD', 'PSAR',
            'BB_upper', 'BB_middle', 'BB_lower', 'Tenkan', 'Kijun', 'SenkouB']

X = df[features].values
# Calculate differences efficiently using NumPy
X_diff = np.diff(X, axis=0, prepend=X[0:1]) # Still good practice, keeps shape consistent
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X_diff)
print(f"Feature engineering and scaling complete. Scaled data shape: {X_scaled.shape}")
print(f"Feature engineering time: {time.time() - start_time:.4f} seconds")


# --- Generate Samples (Vectorized) ---
print("Generating samples (vectorized)...")
start_time = time.time()

data_length = len(X_scaled)
# Indices in the original data where the *future* window starts
# This is consistent with the original loop logic: i starts at window_size
# and goes up to len(X_scaled) - prediction_window - 1
sample_start_indices_in_scaled_data = np.arange(window_size, data_length - prediction_window)
num_samples = len(sample_start_indices_in_scaled_data)

# --- Extract input windows (X) ---
# Use numpy broadcasting or list comprehension + stack for efficient windowing
# This creates indices like [0,1,2..ws-1], [1,2..ws], ...
# Then select rows from X_scaled based on these indices
# A common vectorized way is using as_strided, but list comprehension + stack is often clearer
# and efficient enough for typical window sizes and dataset sizes.
samples = np.stack([X_scaled[i - window_size : i] for i in sample_start_indices_in_scaled_data])

# --- Calculate targets (y) efficiently ---
close_prices = df['Close'].values
# Current price is the price at the end of the input window (index i-1 in original loop,
# which corresponds to index i - 1 relative to sample_start_indices_in_scaled_data if sample starts at i-window_size)
# In our current loop indexing (i from window_size to data_length - prediction_window -1),
# the current price is at index i-1.
# The corresponding index in sample_start_indices_in_scaled_data is i.
# So current price is close_prices[i-1] for sample ending at i-1
current_prices = close_prices[sample_start_indices_in_scaled_data - 1]

# Future prices windows start at index i and end at i + prediction_window - 1
future_prices_windows = np.stack([close_prices[i : i + prediction_window] for i in sample_start_indices_in_scaled_data])
future_means = np.mean(future_prices_windows, axis=1)

# Calculate average returns for all samples
avg_returns = (future_means - current_prices) / current_prices

# Determine targets using boolean indexing (vectorized)
# 1: Hold (default)
# 2: Buy (> threshold)
# 0: Sell (< -threshold)
targets = np.ones(num_samples, dtype=int) # Initialize all as Hold
targets[avg_returns > buy_threshold] = 2  # Set Buy signals
targets[avg_returns < sell_threshold] = 0 # Set Sell signals

print(f"Sample generation complete. Samples shape: {samples.shape}, Targets shape: {targets.shape}")
print(f"Sample generation time: {time.time() - start_time:.4f} seconds")

# --- Torch Dataset ---
print("Creating PyTorch dataset and DataLoader...")
start_time = time.time()
# Convert NumPy arrays to PyTorch tensors and move to device immediately
X_tensor = torch.tensor(samples, dtype=torch.float32).to(device)
y_tensor = torch.tensor(targets, dtype=torch.long).to(device) # Target class indices (0, 1, 2)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Dataset and DataLoader created. Number of batches: {len(loader)}")
print(f"Dataset/DataLoader time: {time.time() - start_time:.4f} seconds")

# --- Model ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3) # 3 classes: Sell, Hold, Buy

    def forward(self, x):
        # h_n, c_n from the last layer
        _, (hn, _) = self.lstm(x)
        # We only need the hidden state of the last layer (num_layers-1)
        # hn has shape (num_layers, batch_size, hidden_size)
        return self.fc(hn[-1])

model = LSTMClassifier(input_size=samples.shape[2], hidden_size=64).to(device) # Move model to device
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # Appropriate for multi-class classification (0, 1, 2)

# --- Train Loop ---
print("Starting training...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # No need to track trading metrics during training batches for speed
    # Calculate overall accuracy at the end of the epoch

    for X_batch, y_batch in loader:
        # X_batch and y_batch are already on the device because TensorDataset data is on device

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted_classes = torch.max(pred, 1)
        correct_predictions += (predicted_classes == y_batch).sum().item()
        total_samples += y_batch.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_samples * 100

    # Removed the complex trading performance calculation from the loop

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

end_time = time.time()
print(f"\nTraining finished in {end_time - start_time:.4f} seconds.")

# --- Optional: Evaluate trading performance AFTER training ---
# This part can be added to evaluate the final model's performance on the training data
# or ideally on a separate validation/test set. It should NOT be inside the training loop.
# Example (evaluating on training data, just for demonstration):
print("\nEvaluating trading performance (on training data)...")
model.eval()
with torch.no_grad():
    # Get predictions for the entire dataset
    full_pred = model(X_tensor) # Use the tensor already on device
    predicted_classes = torch.argmax(full_pred, dim=1).cpu().numpy() # Move back to CPU for NumPy ops

# Map predicted class indices (0, 1, 2) back to signals (-1, 0, 1)
signal_map = {0: -1, 1: 0, 2: 1}
predicted_signals = np.vectorize(signal_map.get)(predicted_classes)

# Calculate returns based on predictions
trade_signals = 0
win_signals = 0
total_return = 0

# Re-calculate current and future prices for vectorized evaluation
# These arrays (current_prices, future_means) were already calculated during sample generation
# We just need the corresponding original price indices.
# The sample generated at index k in the `samples` array corresponds to:
# Input window X_scaled[k : k + window_size]
# Current price close_prices[k + window_size - 1]
# Future prices close_prices[k + window_size : k + window_size + prediction_window]
# Let's use the original indices again. The samples correspond to original data indices
# starting from `window_size` up to `len(X_scaled) - prediction_window - 1`.
# The k-th sample corresponds to original index `k + window_size`.
# The prices needed are `close_prices[k + window_size - 1]` (current) and `close_prices[k + window_size : k + window_size + prediction_window]` (future).
# We have `num_samples` total. The indices for the current prices are `window_size - 1` up to `window_size - 1 + num_samples - 1`.
# The indices for future price means start at `window_size` up to `window_size + num_samples - 1`.

eval_current_prices = close_prices[window_size - 1 : window_size - 1 + num_samples]
eval_future_windows = np.stack([close_prices[i : i + prediction_window]
                                for i in range(window_size, window_size + num_samples)])
eval_future_means = np.mean(eval_future_windows, axis=1)
eval_returns = (eval_future_means - eval_current_prices) / eval_current_prices

# Apply signals to returns
# Only consider trades where signal is not 0 (Hold)
traded_indices = np.where(predicted_signals != 0)[0]

if len(traded_indices) > 0:
    trade_signals = len(traded_indices)
    traded_signals = predicted_signals[traded_indices]
    traded_returns = eval_returns[traded_indices]

    # Winning trades: Buy (1) with positive return OR Sell (-1) with negative return
    winning_trades = np.where((traded_signals == 1) & (traded_returns > 0) |
                              (traded_signals == -1) & (traded_returns < 0))[0]
    win_signals = len(winning_trades)

    # Total return is the sum of absolute returns for winning trades
    # (Assuming winning trades contribute positively, losing trades contribute 0)
    # Or, maybe total return is sum of actual returns for all trades?
    # Let's assume sum of returns for all traded positions as a common metric.
    # If signal is 1, return is +ret. If signal is -1, return is -ret.
    total_return = np.sum(traded_signals * traded_returns) # This sums the PnL% for each trade

    win_rate = (win_signals / trade_signals * 100) if trade_signals > 0 else 0
    avg_trade_return = total_return / trade_signals if trade_signals > 0 else 0 # Average return per trade

    print(f"Evaluation on Training Data:")
    print(f"Total trades: {trade_signals}")
    print(f"Winning trades: {win_signals}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Return (%): {total_return * 100:.4f}%") # Display as percentage
    print(f"Average Return per Trade (%): {avg_trade_return * 100:.4f}%") # Display as percentage
else:
    print("No trade signals generated during evaluation.")