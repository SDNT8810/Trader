import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=50, target_size=1):
        self.data = data
        self.window_size = window_size
        self.target_size = target_size
        
    def __len__(self):
        return len(self.data) - self.window_size - self.target_size + 1
        
    def __getitem__(self, idx):
        # Get input sequence (window of data)
        x = self.data[idx:idx + self.window_size]
        # Get target sequence (next value)
        y = self.data[idx + self.window_size:idx + self.window_size + self.target_size]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=10, dropout=0.2):
        super(TimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Add bidirectional processing
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Deep output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Deep output processing
        output = self.output_layers(context)
        return output

def main():
    # Read the CSV file
    print("Reading NData.csv...")
    df = pd.read_csv('NData.csv')
    
    # Convert to numpy array and normalize
    data = df.values
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Create dataset
    window_size = 50
    num_features = data.shape[1] - 1  # Exclude target column
    dataset = TimeSeriesDataset(data, window_size=window_size)
    
    # Create data loader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model
    model = TimeSeriesModel(
        input_size=num_features,
        hidden_size=256,  # Increased hidden size
        num_layers=10,    # 10 layers
        dropout=0.2
    )
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    
    # Print dataset information
    print(f"\nDataset Information:")
    print(f"Total samples: {len(dataset)}")
    print(f"Window size: {window_size}")
    print(f"Number of features: {num_features}")
    print(f"Batch size: {batch_size}")
    print(f"Number of LSTM layers: {model.num_layers}")
    print(f"Hidden size: {model.hidden_size}")
    print(f"Bidirectional: Yes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    main()
