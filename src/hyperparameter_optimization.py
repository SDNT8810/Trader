import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from train import GoldPricePredictor, load_and_preprocess_data
import logging
import os

def objective(trial):
    """Objective function for hyperparameter optimization."""
    # Define hyperparameters to optimize
    hidden_size = trial.suggest_int('hidden_size', 128, 512)
    num_layers = trial.suggest_int('num_layers', 5, 15)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Load and preprocess data
    X, y = load_and_preprocess_data('data/NData.csv')
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoldPricePredictor(
        input_size=X.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, _ = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_loss

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create study
    study = optuna.create_study(direction='minimize')
    
    # Run optimization
    logging.info('Starting hyperparameter optimization...')
    study.optimize(objective, n_trials=50)
    
    # Print results
    logging.info('Best trial:')
    trial = study.best_trial
    logging.info(f'Value: {trial.value:.6f}')
    logging.info('Params:')
    for key, value in trial.params.items():
        logging.info(f'    {key}: {value}')
    
    # Save results
    os.makedirs('checkpoints/optimization', exist_ok=True)
    results_df = study.trials_dataframe()
    results_df.to_csv('checkpoints/optimization/optimization_results.csv', index=False)
    
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image('Figs/optimization_history.png')
    
    # Plot parameter importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image('Figs/parameter_importance.png')
    
    logging.info('Hyperparameter optimization completed!')

if __name__ == '__main__':
    main() 