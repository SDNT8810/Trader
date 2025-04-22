import os
import sys
import torch
import numpy as np
from pathlib import Path
import yaml
import pandas as pd

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data.data_preprocessor import DataPreprocessor
from data.cost_function import CostFunction
from models.trading_model import TradingANN, ModelTrainer
from visualization.training_monitor import TrainingMonitor

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("1. Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.process_data()
    
    if data is None:
        print("Error: Failed to process data")
        return
    
    print("\n2. Creating ANN Model...")
    config = load_config()
    model = TradingANN(config)
    
    print("\n3. Training Model...")
    trainer = ModelTrainer(model, config)
    monitor = TrainingMonitor(config_path='config/config.yaml')
    
    # Split data into train and validation sets
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"Train set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    
    # Train the model
    trainer.train(train_data, val_data, monitor)
    
    print("\n4. Training completed!")
    print("Check the training progress in src/Figs/Training/training_progress.html")

if __name__ == "__main__":
    main()
