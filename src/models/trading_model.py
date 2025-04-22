import torch
import torch.nn as nn
import yaml
import os
from torch.utils.data import DataLoader
import numpy as np
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class TradingANN(nn.Module):
    def __init__(self, config):
        super(TradingANN, self).__init__()
        
        # Store config
        self.config = config
        
        # Get model parameters from config
        model_config = config['model']
        self.window_size = model_config['window_size']
        self.num_features = model_config['num_features']
        self.prediction_window = model_config['prediction_window']
        
        # First layer processes 3D input (batch_size, window_size, num_features)
        self.first_layer = nn.Sequential(
            nn.Conv1d(self.num_features, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Calculate size after first layer and flattening
        self.flattened_size = 128 * self.window_size
        
        # Main network layers
        self.layers = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)  # Single output for trading signal
        )
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize criterion and optimizer
        self.criterion = nn.MSELoss()
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs(model_config['save_dir'], exist_ok=True)
    
    def prepare_data(self, data):
        """Convert pandas DataFrame to PyTorch Dataset"""
        from torch.utils.data import TensorDataset
        
        # Create sequences using preprocessor
        X, y = self.preprocessor.create_sequences(data)
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Reshape X to (batch_size, window_size * num_features)
        batch_size = X.shape[0]
        X = X.reshape(batch_size, -1)
        
        return TensorDataset(X, y)

    def forward(self, x):
        # x shape: (batch_size, window_size, num_features)
        batch_size = x.size(0)
        
        # Permute for Conv1d: (batch_size, num_features, window_size)
        x = x.permute(0, 2, 1)
        
        # Process through first layer
        x = self.first_layer(x)
        
        # Flatten for the main network
        x = x.reshape(batch_size, -1)
        
        # Pass through main network
        output = self.layers(x)
        
        # Apply tanh to get output between -1 and 1
        return torch.tanh(output)
    
    def count_parameters(self):
        """Count total number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize criterion and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                         lr=self.config['model']['learning_rate'],
                                         weight_decay=0.001,
                                         betas=(0.9, 0.999))
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=True
        )
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs(self.config['model']['save_dir'], exist_ok=True)
        
        # Load data preprocessor to get scalers
        from data.data_preprocessor import DataPreprocessor
        self.preprocessor = DataPreprocessor()
        self.preprocessor.process_data()
        self.mean_price_scaler = self.preprocessor.scalers['Mean_Price']
    
    def prepare_data(self, data):
        """Convert pandas DataFrame to PyTorch Dataset"""
        from torch.utils.data import TensorDataset
        
        # Create sequences using preprocessor
        X, y = self.preprocessor.create_sequences(data)
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        return TensorDataset(X, y)
    
    def calculate_trading_metrics(self, signals, future_prices, current_prices):
        """Calculate trading performance metrics"""
        # Convert signals to trading decisions (-1: sell, 0: hold, 1: buy)
        decisions = torch.sign(signals)
        
        # Calculate potential returns
        returns = torch.zeros_like(signals)
        for i in range(len(signals)):
            if decisions[i] > 0:  # Buy signal
                # Calculate return based on future price movement
                future_return = (future_prices[i] - current_prices[i]) / current_prices[i]
                returns[i] = future_return
            elif decisions[i] < 0:  # Sell signal
                # Calculate return based on future price movement
                future_return = (current_prices[i] - future_prices[i]) / current_prices[i]
                returns[i] = future_return
        
        # Calculate metrics
        accuracy = torch.mean((returns > 0).float()).item() * 100  # Percentage of profitable trades
        avg_return = torch.mean(returns).item() * 100  # Average return in percentage
        win_rate = torch.mean((returns > 0).float()).item() * 100  # Win rate
        avg_win = torch.mean(returns[returns > 0]).item() * 100 if torch.sum(returns > 0) > 0 else 0
        avg_loss = torch.mean(returns[returns < 0]).item() * 100 if torch.sum(returns < 0) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'avg_return': avg_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def custom_loss(self, signals, targets, future_prices, current_prices):
        """Custom loss function that evaluates trading performance"""
        # Convert signals to trading decisions
        decisions = torch.sign(signals)
        
        # Calculate potential returns
        returns = torch.zeros_like(signals)
        for i in range(len(signals)):
            if decisions[i] > 0:  # Buy signal
                future_return = (future_prices[i] - current_prices[i]) / current_prices[i]
                returns[i] = future_return
            elif decisions[i] < 0:  # Sell signal
                future_return = (current_prices[i] - future_prices[i]) / current_prices[i]
                returns[i] = future_return
        
        # Calculate loss components
        mse_loss = self.criterion(signals, targets)  # MSE between predicted and target signals
        return_loss = -torch.mean(returns)  # Negative mean return (we want to maximize returns)
        
        # Add regularization for signal magnitude
        signal_magnitude_loss = torch.mean(torch.abs(signals))
        
        # Combine losses with weights
        total_loss = 0.2 * mse_loss + 0.7 * return_loss + 0.1 * signal_magnitude_loss
        
        return total_loss
    
    def train(self, train_data, val_data, monitor):
        """Train the model with trading signal prediction"""
        # Prepare datasets
        train_dataset = self.prepare_data(train_data)
        val_dataset = self.prepare_data(val_data)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, 
                                batch_size=self.config['model']['batch_size'], 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, 
                              batch_size=self.config['model']['batch_size'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        min_delta = self.config['model']['min_delta']
        patience = self.config['model']['patience']
        
        print("\nTraining Started:")
        print("=" * 100)
        print("Epoch | Train Loss | Val Loss | Accuracy | Win Rate | Avg Return | Time")
        print("-" * 100)
        
        for epoch in range(self.config['model']['epochs']):
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_returns = []
            train_win_count = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get current and future prices
                current_prices = inputs[:, -1, 0]  # Last price in the window
                future_prices = targets  # Target is now the trading signal
                
                # Forward pass
                signals = self.model(inputs)
                
                # Calculate loss using custom loss function
                loss = self.custom_loss(signals.squeeze(), targets, future_prices, current_prices)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy and returns
                predicted_direction = torch.sign(signals.squeeze())
                actual_direction = torch.sign(targets)
                train_correct += (predicted_direction == actual_direction).sum().item()
                train_total += targets.size(0)
                
                # Calculate returns and win rate
                returns = targets * predicted_direction
                train_returns.extend(returns.detach().cpu().numpy())
                train_win_count += (returns > 0).sum().item()
            
            # Calculate training metrics
            train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            train_returns = np.array(train_returns)
            train_win_rate = 100 * train_win_count / train_total
            train_avg_return = np.mean(train_returns) * 100
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_returns = []
            val_win_count = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Get current and future prices
                    current_prices = inputs[:, -1, 0]
                    future_prices = targets
                    
                    # Forward pass
                    signals = self.model(inputs)
                    
                    # Calculate loss
                    loss = self.custom_loss(signals.squeeze(), targets, future_prices, current_prices)
                    val_loss += loss.item()
                    
                    # Calculate accuracy and returns
                    predicted_direction = torch.sign(signals.squeeze())
                    actual_direction = torch.sign(targets)
                    val_correct += (predicted_direction == actual_direction).sum().item()
                    val_total += targets.size(0)
                    
                    # Calculate returns and win rate
                    returns = targets * predicted_direction
                    val_returns.extend(returns.detach().cpu().numpy())
                    val_win_count += (returns > 0).sum().item()
            
            # Calculate validation metrics
            val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            val_returns = np.array(val_returns)
            val_win_rate = 100 * val_win_count / val_total
            val_avg_return = np.mean(val_returns) * 100
            
            epoch_time = time.time() - epoch_start_time
            
            # Update training monitor
            monitor.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                train_win_rate=train_win_rate,
                val_win_rate=val_win_rate,
                train_avg_return=train_avg_return,
                val_avg_return=val_avg_return,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                gradient_norm=torch.norm(torch.cat([p.grad.view(-1) for p in self.model.parameters()])).item(),
                batch_size=self.config['model']['batch_size'],
                num_samples=len(train_dataset)
            )
            
            # Print progress
            print(f"{epoch+1:5d} | {train_loss:.6f} | {val_loss:.6f} | {val_accuracy:.2f}% | "
                  f"{val_win_rate:.2f}% | {val_avg_return:.2f}% | {epoch_time:.2f}s")
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Save best model based on validation loss
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(f"best_model_epoch_{epoch+1}.pt")
                print(f"New best model saved! (val_loss: {val_loss:.6f}, accuracy: {val_accuracy:.2f}%)")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print("\nEarly stopping triggered - No improvement in validation loss")
                break
        
        print("\nTraining Completed!")
        print("=" * 100)
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final Accuracy: {val_accuracy:.2f}%")
        print(f"Final Win Rate: {val_win_rate:.2f}%")
        print(f"Final Average Return: {val_avg_return:.2f}%")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, os.path.join(self.config['model']['save_dir'], filename))

def create_model(config_path='config/config.yaml'):
    model = TradingANN(config_path)
    print(f"Model Architecture:")
    print(f"Input size: {model.flattened_size} (W={model.window_size} * N={model.num_features})")
    print(f"Hidden layers: {[model.layers[i].out_features for i in range(0, len(model.layers), 3)]}")
    print(f"Output size: {model.prediction_window} (M mean prices)")
    print(f"Total parameters: {model.count_parameters():,}")
    return model

def calculate_mae(self, predictions, targets):
    """Calculate Mean Absolute Error"""
    return torch.mean(torch.abs(predictions - targets)).item()

def calculate_mse(self, predictions, targets):
    """Calculate Mean Squared Error"""
    return torch.mean((predictions - targets) ** 2).item()

def calculate_accuracy(self, predictions, targets, tolerance=0.01):
    """Calculate prediction accuracy within tolerance"""
    correct = torch.abs(predictions - targets) <= tolerance
    return torch.mean(correct.float()).item()
