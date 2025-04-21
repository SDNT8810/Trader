import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

class RiskManager:
    def __init__(self, account_size: float):
        self.account_size = account_size
        self.max_position_size = 0.02 * account_size
        self.daily_loss_limit = 0.05 * account_size
        self.max_drawdown = 0.15 * account_size
        self.current_drawdown = 0.0
        self.daily_loss = 0.0
        
    def check_position_size(self, position: float) -> bool:
        return position <= self.max_position_size
        
    def check_daily_loss(self, loss: float) -> bool:
        self.daily_loss += loss
        return self.daily_loss <= self.daily_loss_limit
        
    def check_drawdown(self, current_value: float) -> bool:
        self.current_drawdown = max(self.current_drawdown, 
                                  (self.account_size - current_value) / self.account_size)
        return self.current_drawdown <= self.max_drawdown
        
    def reset_daily_metrics(self):
        self.daily_loss = 0.0

class FeatureEngineer:
    def __init__(self, lags: List[int] = [1, 5, 10, 20], 
                 windows: List[int] = [5, 10, 20]):
        self.lags = lags
        self.windows = windows
        
    def add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            for lag in self.lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df
        
    def add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            for window in self.windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                df[f'{col}_rolling_corr_{window}'] = df[col].rolling(window).corr()
        return df
        
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['hour'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        return df

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int = 50, target_size: int = 1):
        self.data = data
        self.window_size = window_size
        self.target_size = target_size
        
    def __len__(self) -> int:
        return len(self.data) - self.window_size - self.target_size + 1
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.target_size]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 num_layers: int = 10, dropout: float = 0.2):
        super(TimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with residual connections
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            dropout=dropout
        )
        
        # Deep output network with skip connections
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)  # Skip connection
        
        # Output prediction
        output = self.output_net(attn_out[:, -1, :])
        return output

class ModelTrainer:
    def __init__(self, model: nn.Module, risk_manager: RiskManager):
        self.model = model
        self.risk_manager = risk_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, learning_rate: float = 0.001):
        criterion = nn.HuberLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.3
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Check risk management
                if not self.risk_manager.check_daily_loss(loss.item()):
                    logging.warning("Daily loss limit reached")
                    break
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self.validate(val_loader, criterion)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info("Early stopping triggered")
                    break
            
            logging.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f}")
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        return val_loss / len(val_loader)

def main():
    # Read and preprocess data
    logging.info("Reading NData.csv...")
    df = pd.read_csv('NData.csv', index_col=0)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.add_lagged_features(df)
    df = feature_engineer.add_rolling_stats(df)
    df = feature_engineer.add_time_features(df)
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Normalize data
    data = df.values
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Create dataset
    window_size = 50
    num_features = data.shape[1] - 1
    dataset = TimeSeriesDataset(data, window_size=window_size)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and risk manager
    model = TimeSeriesModel(
        input_size=num_features,
        hidden_size=256,
        num_layers=10,
        dropout=0.2
    )
    
    risk_manager = RiskManager(account_size=100000.0)  # Example account size
    
    # Train model
    trainer = ModelTrainer(model, risk_manager)
    trainer.train(train_loader, val_loader, num_epochs=100)
    
    # Print model summary
    logging.info("\nModel Architecture:")
    logging.info(model)
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    main()
