import pandas as pd
import numpy as np
import talib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the CSV file
df = pd.read_csv('Gchart.csv')

# Create indicators using TA-Lib
def add_indicators(df):
    df['Ichimoku_Tenkan'] = talib.ICHIMOKU(df['High'], df['Low'], df['Close'], 9, 26, 52)[0]
    df['Ichimoku_Kijun'] = talib.ICHIMOKU(df['High'], df['Low'], df['Close'], 9, 26, 52)[1]
    df['Ichimoku_SpanB'] = talib.ICHIMOKU(df['High'], df['Low'], df['Close'], 9, 26, 52)[2]
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], 12, 26, 9)
    df['PSAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
    df['EMA50'] = talib.EMA(df['Close'], timeperiod=50)
    df['EMA200'] = talib.EMA(df['Close'], timeperiod=200)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    return df

# Apply the indicator calculations
df = add_indicators(df)

# Create the dataset
window_size = 50
def create_features(df, window_size=50, N=10):
    X = []
    y = []
    for i in range(window_size, len(df) - N):
        window = df.iloc[i - window_size:i]
        features = window[['High', 'Open', 'Close', 'Low', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_SpanB', 'MACD', 'MACD_signal', 'PSAR', 'EMA50', 'EMA200', 'Upper_BB', 'Middle_BB', 'Lower_BB']].values
        target = np.sign(df['Close'].iloc[i + N] - df['Close'].iloc[i])  # +1 for buy, -1 for sell
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y)

# Create features and target variables
X, y = create_features(df, window_size=window_size)

# Normalize the features (price-based derivatives)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = np.array([scaler.fit_transform(x) for x in X])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(window_size, X.shape[2])),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='tanh')  # Tanh to output values between -1 and 1
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.sign(y_pred)  # +1 for buy, -1 for sell

# Evaluate the reward/cost function (accuracy of decisions)
def calculate_rewards(df, window_size, model, N=10):
    rewards = []
    initial_balance = 100
    balance = initial_balance
    position = 0  # 1 for buy, -1 for sell, 0 for hold
    for i in range(window_size, len(df) - N):
        window = df.iloc[i - window_size:i]
        features = window[['High', 'Open', 'Close', 'Low', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_SpanB', 'MACD', 'MACD_signal', 'PSAR', 'EMA50', 'EMA200', 'Upper_BB', 'Middle_BB', 'Lower_BB']].values
        features_scaled = scaler.transform(features)
        action = np.sign(model.predict(features_scaled.reshape(1, window_size, 15)))
        
        # Buy
        if action == 1 and position == 0:
            buy_price = df['Close'].iloc[i]
            position = 1  # We bought
        # Sell
        elif action == -1 and position == 1:
            sell_price = df['Close'].iloc[i]
            profit = sell_price - buy_price
            balance += profit
            position = 0  # We sold
            
        rewards.append(balance)
    
    return rewards

# Simulate trading using the model
rewards = calculate_rewards(df, window_size=window_size, model=model)
final_balance = rewards[-1]
print(f'Final Balance after 240 candles: {final_balance}')