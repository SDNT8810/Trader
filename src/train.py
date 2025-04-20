import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class GoldPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=4, dropout=0.2):
        super(GoldPricePredictor, self).__init__()
        
        # LSTM layers with residual connections
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Attention mechanism with multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,  # Reduced number of heads
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
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)  # Skip connection
        
        # Output prediction
        output = self.output_net(attn_out[:, -1, :])  # Take last time step
        return output, attn_out

class TrainingMonitor:
    def __init__(self, save_dir):
        self.train_losses = []
        self.val_losses = []
        self.returns = []
        self.sharpe_ratios = []
        self.max_drawdowns = []
        self.accuracy = []
        self.best_val_loss = float('inf')
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Training plots will be saved to: {save_dir}")
    
    def update(self, train_loss, val_loss, predictions, actuals):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Calculate returns
        returns = np.diff(predictions) / predictions[:-1]
        self.returns.append(returns)
        
        # Calculate Sharpe ratio
        sharpe = np.mean(returns) / np.std(returns) if len(returns) > 0 else 0
        self.sharpe_ratios.append(sharpe)
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        self.max_drawdowns.append(max_drawdown)
        
        # Calculate accuracy (direction prediction)
        correct_direction = np.sign(np.diff(predictions)) == np.sign(np.diff(actuals))
        accuracy = np.mean(correct_direction) if len(correct_direction) > 0 else 0
        self.accuracy.append(accuracy)
        
        self.plot_progress()
    
    def plot_progress(self):
        try:
            plt.figure(figsize=(15, 10))
            
            # Loss plot
            plt.subplot(2, 2, 1)
            plt.plot(self.train_losses, label='Train Loss')
            plt.plot(self.val_losses, label='Val Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            # Returns plot
            plt.subplot(2, 2, 2)
            if self.returns:
                plt.plot(self.returns[-1], label='Returns')
            plt.title('Returns Distribution')
            plt.legend()
            
            # Sharpe ratio and max drawdown
            plt.subplot(2, 2, 3)
            plt.plot(self.sharpe_ratios, label='Sharpe Ratio')
            plt.plot(self.max_drawdowns, label='Max Drawdown')
            plt.title('Risk Metrics')
            plt.legend()
            
            # Accuracy
            plt.subplot(2, 2, 4)
            plt.plot(self.accuracy, label='Direction Accuracy')
            plt.title('Prediction Accuracy')
            plt.legend()
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.save_dir, 'training_progress.png')
            plt.savefig(plot_path)
            plt.close()
            
            logging.info(f"Plot saved to: {plot_path}")
            
        except Exception as e:
            logging.error(f"Error saving plot: {str(e)}")
            plt.close()

def load_and_preprocess_data(file_path, sequence_length=50):
    """Load and preprocess the data for training."""
    # Load data
    df = pd.read_csv(file_path)
    
    # Print data structure
    print("Data columns:", df.columns.tolist())
    print("Data shape:", df.shape)
    
    # Extract features and target
    feature_columns = [col for col in df.columns if col not in ['Date', 'Close', 'Index', 'Unnamed: 0']]
    if not feature_columns:
        feature_columns = df.columns[:-1]  # Use all columns except the last one as features
    
    features = df[feature_columns].values
    target = df['Close'].values if 'Close' in df.columns else df.iloc[:, -1].values
    
    # Handle infinite values
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    target = np.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Remove any remaining problematic values
    features = np.clip(features, -1e6, 1e6)
    target = np.clip(target, -1e6, 1e6)
    
    # Print value ranges
    print("Features range:", np.min(features), "to", np.max(features))
    print("Target range:", np.min(target), "to", np.max(target))
    
    # Normalize features using robust scaling
    # Scale features
    feature_scaler = RobustScaler()
    features_normalized = feature_scaler.fit_transform(features)
    
    # Scale target
    target_scaler = RobustScaler()
    target_normalized = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_normalized) - sequence_length):
        X.append(features_normalized[i:(i + sequence_length)])
        y.append(target_normalized[i + sequence_length])
    
    return np.array(X), np.array(y), target_scaler  # Return the target scaler for later denormalization

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, num_epochs, checkpoint_dir, target_scaler):
    """Train the model with early stopping and checkpointing."""
    monitor = TrainingMonitor(os.path.join('src', 'Figs', 'Training'))
    best_val_loss = float('inf')
    patience = 15  # Increased patience
    patience_counter = 0
    
    # Learning rate scheduler with adjusted parameters
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,  # Increased maximum learning rate
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.2,  # Reduced warmup period
        anneal_strategy='cos',  # Cosine annealing
        div_factor=10.0,  # Initial learning rate = max_lr/10
        final_div_factor=100.0  # Final learning rate = max_lr/1000
    )
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        all_predictions = []
        all_actuals = []
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            
            # Huber loss for robustness
            loss = criterion(outputs.squeeze(), batch_y)
            
            # L2 regularization with reduced weight
            l2_lambda = 0.0005  # Reduced regularization
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            
            # Gradient clipping with increased threshold
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Store predictions and actuals
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_actuals.extend(batch_y.cpu().numpy())
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, _ = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
                
                # Store validation predictions and actuals
                val_predictions.extend(outputs.cpu().numpy())
                val_actuals.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Denormalize predictions and actuals
        val_predictions = target_scaler.inverse_transform(np.array(val_predictions).reshape(-1, 1)).flatten()
        val_actuals = target_scaler.inverse_transform(np.array(val_actuals).reshape(-1, 1)).flatten()
        
        # Update monitor
        monitor.update(train_loss, val_loss, val_predictions, val_actuals)
        
        # Log progress
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        logging.info(f'Sharpe Ratio: {monitor.sharpe_ratios[-1]:.4f}')
        logging.info(f'Max Drawdown: {monitor.max_drawdowns[-1]:.4f}')
        logging.info(f'Direction Accuracy: {monitor.accuracy[-1]:.4f}')
        logging.info(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'sharpe_ratio': monitor.sharpe_ratios[-1],
                'max_drawdown': monitor.max_drawdowns[-1],
                'accuracy': monitor.accuracy[-1]
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info('Early stopping triggered')
                break

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join('checkpoints', timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create training plots directory
    os.makedirs(os.path.join('src', 'Figs', 'Training'), exist_ok=True)
    
    # Load and preprocess data
    logging.info('Loading and preprocessing data...')
    X, y, target_scaler = load_and_preprocess_data('NData.csv')
    
    # Save the scaler
    joblib.dump(target_scaler, os.path.join(checkpoint_dir, 'target_scaler.pkl'))
    
    # Print data shapes
    logging.info(f'X shape: {X.shape}')
    logging.info(f'y shape: {y.shape}')
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = 128  # Increased batch size for more stable training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_train.shape[2]  # Number of features
    model = GoldPricePredictor(input_size).to(device)
    
    # Print model summary
    logging.info(f'Model input size: {input_size}')
    logging.info(f'Model architecture:\n{model}')
    
    # Define loss function (Huber loss for robustness)
    criterion = nn.HuberLoss()
    
    # Define optimizer with adjusted weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.01,  # Increased initial learning rate
        weight_decay=0.005,  # Reduced weight decay
        betas=(0.9, 0.999)
    )
    
    # Train model
    logging.info('Starting training...')
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=100,
        checkpoint_dir=checkpoint_dir,
        target_scaler=target_scaler
    )
    
    logging.info('Training completed!')

if __name__ == '__main__':
    main() 