import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import os
from typing import Tuple, Dict, List, Any
from models.trading_model import TradingANN, create_model
from model_evaluation import ModelEvaluator
from visualization.training_monitor import TrainingMonitor
import multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method, Process
import platform
import psutil
from tqdm import tqdm
import logging
import time
import shutil
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from report_scheduler import ReportScheduler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Create checkpoint directory
CHECKPOINT_DIR = 'checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Force spawn method for better process isolation
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Get number of physical cores
def get_physical_cores():
    if platform.system() == 'Darwin':  # macOS
        return psutil.cpu_count(logical=False)
    else:
        return psutil.cpu_count(logical=False)

# Set number of workers based on physical cores
NUM_WORKERS = get_physical_cores()  # Use all physical cores
print(f"Using {NUM_WORKERS} physical cores for training")

# Default training configuration
DEFAULT_TRAINING_CONFIG = {
    'training': {
        'batch_size': 128,
        'learning_rate': 0.0005,
        'epochs': 2000,
        'optimizer': {
            'type': 'AdamW',
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            'weight_decay': 0.001,
            'amsgrad': False
        },
        'loss_function': {
            'type': 'huber',
            'delta': 1.0
        },
        'device': 'auto',  # 'auto', 'cpu', 'cuda', or 'mps'
        'plot_frequency': 5,
        'gradient_clip': 0.5,
        'gradient_accumulation_steps': 2,
        'early_stopping': {
            'patience': 50,
            'min_delta': 0.0001
        },
        'checkpoint': {
            'save_every': 25,
            'keep_last': 10
        },
        'scheduler': {
            'type': 'cosine_warmup',  # 'cosine_warmup', 'reduce_on_plateau', 'one_cycle'
            'warmup_epochs': 3,
            'min_lr': 1e-6,
            'patience': 5,
            'factor': 0.5
        }
    },
    'data': {
        'input_shape': [40, 43],
        'output_shape': [11],
        'normalization': True,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'shuffle': True,
        'pin_memory': True,
        'num_workers': 0,
        'persistent_workers': False
    }
}

# Set device
if platform.system() == 'Darwin':  # macOS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for M1/M2 GPU")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
elif platform.system() == 'Windows':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Configure PyTorch for multi-core processing
torch.set_num_threads(1)  # Use single thread per process
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

class TimeSeriesDataset(Dataset):
    """Dataset for time series data with sliding window"""
    
    def __init__(self, input_data: np.ndarray, target_data: np.ndarray, window_size: int):
        self.input_data = input_data.astype(np.float32)  # Ensure float32 type
        self.target_data = target_data.astype(np.float32)  # Ensure float32 type
        self.window_size = window_size
        self.input_shape = input_data.shape[1]  # Store input shape for reshaping
        
    def __len__(self) -> int:
        return len(self.input_data) - self.window_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get input window
        x = self.input_data[idx:idx + self.window_size]
        # Keep the window shape (window_size, features) for LSTM input
        x = x.reshape(self.window_size, self.input_shape)
        # Get target (next time step's price changes)
        y = self.target_data[idx + self.window_size]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """Load configuration from YAML file with fallback to defaults"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults to ensure all parameters exist
        merged_config = DEFAULT_TRAINING_CONFIG.copy()
        
        # Update training config
        if 'training' in config:
            deep_update_dict(merged_config['training'], config['training'])
        
        # Update data config
        if 'data' in config:
            deep_update_dict(merged_config['data'], config['data'])
        
        return merged_config
    except Exception as e:
        logging.warning(f"Error loading config file: {str(e)}")
        logging.warning("Using default configuration.")
        return DEFAULT_TRAINING_CONFIG

def deep_update_dict(d: Dict, u: Dict) -> Dict:
    """Recursively update dictionary with another dictionary"""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            deep_update_dict(d[k], v)
        else:
            d[k] = v
    return d

def get_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Create optimizer based on configuration"""
    optimizer_type = config['training']['optimizer'].lower()
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.01)
    optimizer_config = config.get('optimizer', {})
    
    if optimizer_type == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(optimizer_config.get('beta1', 0.9), 
                  optimizer_config.get('beta2', 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(optimizer_config.get('beta1', 0.9), 
                  optimizer_config.get('beta2', 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=optimizer_config.get('nesterov', False)
        )
    else:
        # Default to AdamW if unknown optimizer type
        logging.warning(f"Unknown optimizer type: {optimizer_type}. Using AdamW.")
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

def get_loss_function(config: Dict) -> nn.Module:
    """Create loss function based on configuration"""
    loss_type = config['training']['loss_function'].lower()
    
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'huber':
        return nn.HuberLoss(delta=1.0)  # Using default delta
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'smoothl1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")

def get_scheduler(optimizer: optim.Optimizer, config: Dict, num_batches: int) -> Any:
    """Create learning rate scheduler based on configuration"""
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosineannealingwarmrestarts').lower()
    
    if scheduler_type == 'cosineannealingwarmrestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 100),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            min_lr=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'one_cycle':
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training']['learning_rate'],
            epochs=config['training']['epochs'],
            steps_per_epoch=num_batches,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    else:
        # Default to CosineAnnealingWarmRestarts if unknown scheduler type
        logging.warning(f"Unknown scheduler type: {scheduler_type}. Using CosineAnnealingWarmRestarts.")
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=2,
            eta_min=1e-6
        )

def get_device(config: Dict) -> torch.device:
    """Get the appropriate device based on configuration"""
    device_preference = config['training'].get('device', 'auto').lower()
    
    if device_preference == 'auto':
        if platform.system() == 'Darwin' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    elif device_preference == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_preference == 'mps' and platform.system() == 'Darwin' and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                   train_loss: float, val_loss: float, is_best: bool = False) -> None:
    """Save model checkpoint"""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        torch.save(state, best_path)
        logging.info(f"New best model saved at epoch {epoch}")

def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, 
                   checkpoint_path: str) -> Tuple[int, float, float]:
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return (
        checkpoint['epoch'],
        checkpoint['train_loss'],
        checkpoint['val_loss']
    )

def train_worker(worker_id: int, config: Dict, device: torch.device, 
                epochs_per_worker: int, start_epoch: int, report_scheduler: ReportScheduler) -> None:
    """Worker process for training"""
    try:
        # Set process affinity to specific core
        if platform.system() == 'Linux':
            os.sched_setaffinity(0, {worker_id})
        
        # Initialize training monitor
        monitor = TrainingMonitor(save_dir='src/Figs/Training')
        
        # Submit initialization report
        report_scheduler.submit_report(worker_id, 'init', {
            'device_type': device.type,
            'worker_id': worker_id,
            'epochs_range': f"{start_epoch}-{start_epoch + epochs_per_worker}"
        })
        
        # Load data for this worker
        normalized_data = pd.read_csv('NData.csv')
        input_data = normalized_data.iloc[:, 1:-2].values
        output_data = pd.read_csv('Output.csv')
        target_data = output_data.iloc[:, :-1].values
        
        # Create dataset
        window_size = config['data']['input_shape'][0]
        num_features = config['data']['input_shape'][1]
        dataset = TimeSeriesDataset(input_data, target_data, window_size)
        
        # Split into train and validation
        train_size = int(config['data']['train_ratio'] * len(dataset))
        val_size = int(config['data']['val_ratio'] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders with memory optimization
        batch_size = config['training']['batch_size']
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=config['data']['shuffle'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            persistent_workers=config['data']['persistent_workers']
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            persistent_workers=config['data']['persistent_workers']
        )
        
        # Create and move model to device
        model = create_model(config_path='config/config.yaml').to(device)
        
        # Load best model if it exists
        best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        if os.path.exists(best_model_path):
            try:
                report_scheduler.submit_report(worker_id, 'info', {
                    'message': f"Loading best model from {best_model_path}"
                })
                checkpoint = torch.load(best_model_path)
                
                # Check if model architectures match
                current_state_dict = model.state_dict()
                checkpoint_state_dict = checkpoint['state_dict']
                
                # Check if keys match
                if set(current_state_dict.keys()) != set(checkpoint_state_dict.keys()):
                    report_scheduler.submit_report(worker_id, 'warning', {
                        'message': "Model architectures don't match. Starting fresh training."
                    })
                else:
                    # Check if shapes match
                    shapes_match = True
                    for key in current_state_dict:
                        if current_state_dict[key].shape != checkpoint_state_dict[key].shape:
                            shapes_match = False
                            break
                    
                    if shapes_match:
                        model.load_state_dict(checkpoint['state_dict'])
                        report_scheduler.submit_report(worker_id, 'info', {
                            'message': "Successfully loaded model checkpoint"
                        })
                    else:
                        report_scheduler.submit_report(worker_id, 'warning', {
                            'message': "Model parameter shapes don't match. Starting fresh training."
                        })
            except Exception as e:
                report_scheduler.submit_report(worker_id, 'error', {
                    'message': f"Error loading checkpoint: {str(e)}"
                })
        
        # Get training components from config
        criterion = get_loss_function(config)
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config, len(train_loader))
        
        # Initialize mixed precision training
        scaler = GradScaler()
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
        
        # Track best validation loss
        best_val_loss = float('inf')
        gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
        
        try:
            for epoch in range(start_epoch, start_epoch + epochs_per_worker):
                # Training phase
                model.train()
                train_loss = 0.0
                train_accuracy = 0.0
                train_predictions = []
                train_targets = []
                total_norm = 0.0
                count_norm = 0
                total_samples = 0
                
                optimizer.zero_grad()
                
                # Submit training start report
                report_scheduler.submit_report(worker_id, 'train_start', {
                    'epoch': epoch + 1,
                    'total_batches': len(train_loader)
                })
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    try:
                        # Move data to device with non-blocking transfer and ensure correct shape
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        
                        # Mixed precision training
                        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                            # Ensure inputs have correct shape before forward pass
                            batch_size = inputs.size(0)
                            # Reshape to (batch_size, window_size, input_dim)
                            window_size = config['data']['input_shape'][0]
                            input_dim = config['data']['input_shape'][1]
                            inputs = inputs.view(batch_size, window_size, input_dim)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            
                            # Add L2 regularization if configured
                            if 'l2_lambda' in config['training']:
                                l2_lambda = config['training']['l2_lambda']
                                l2_reg = torch.tensor(0.).to(device)
                                for param in model.parameters():
                                    l2_reg += torch.norm(param)
                                loss += l2_lambda * l2_reg
                        
                        # Scale loss and backpropagate
                        scaler.scale(loss).backward()
                        
                        # Gradient accumulation
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            # Unscale gradients for clipping
                            scaler.unscale_(optimizer)
                            
                            # Calculate gradient norm
                            total_norm += torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                max_norm=config['training']['gradient_clip']
                            ).item()
                            count_norm += 1
                            
                            # Update weights
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            
                            # Update learning rate
                            if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                                scheduler.step()
                        
                        train_loss += loss.item() * inputs.size(0)
                        
                        # Calculate accuracy
                        with torch.no_grad():
                            predictions = outputs.detach()
                            correct_direction = torch.sign(predictions) == torch.sign(targets)
                            accuracy = torch.mean(correct_direction.float())
                            train_accuracy += accuracy.item() * inputs.size(0)
                        
                        total_samples += inputs.size(0)
                        
                        # Store predictions and targets
                        train_predictions.append(outputs.detach().cpu().numpy())
                        train_targets.append(targets.cpu().numpy())
                        
                        # Submit training update report
                        if (batch_idx + 1) % config['training']['plot_frequency'] == 0:
                            report_scheduler.submit_report(worker_id, 'train_update', {
                                'loss': f'{loss.item():.6f}',
                                'acc': f'{accuracy.item():.4f}',
                                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                            })
                        
                    except Exception as e:
                        report_scheduler.submit_report(worker_id, 'error', {
                            'message': f"Error in training batch {batch_idx + 1}: {str(e)}"
                        })
                        raise
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_accuracy = 0.0
                val_predictions = []
                val_targets = []
                val_samples = 0
                
                # Submit validation start report
                report_scheduler.submit_report(worker_id, 'val_start', {
                    'epoch': epoch + 1,
                    'total_batches': len(val_loader)
                })
                
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(val_loader):
                        try:
                            # Move data to device with non-blocking transfer
                            inputs = inputs.to(device, non_blocking=True)
                            targets = targets.to(device, non_blocking=True)
                            
                            # Reshape to (batch_size, window_size, input_dim)
                            batch_size = inputs.size(0)
                            window_size = config['data']['input_shape'][0]
                            input_dim = config['data']['input_shape'][1]
                            inputs = inputs.view(batch_size, window_size, input_dim)
                            
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item() * inputs.size(0)
                            
                            # Calculate accuracy
                            predictions = outputs
                            correct_direction = torch.sign(predictions) == torch.sign(targets)
                            accuracy = torch.mean(correct_direction.float())
                            val_accuracy += accuracy.item() * inputs.size(0)
                            
                            val_samples += inputs.size(0)
                            
                            # Store predictions and targets
                            val_predictions.append(outputs.cpu().numpy())
                            val_targets.append(targets.cpu().numpy())
                            
                            # Submit validation update report
                            if (batch_idx + 1) % config['training']['plot_frequency'] == 0:
                                report_scheduler.submit_report(worker_id, 'val_update', {
                                    'loss': f'{loss.item():.6f}',
                                    'acc': f'{accuracy.item():.4f}'
                                })
                            
                        except Exception as e:
                            report_scheduler.submit_report(worker_id, 'error', {
                                'message': f"Error in validation batch {batch_idx + 1}: {str(e)}"
                            })
                            raise
                
                # Calculate average losses and metrics
                avg_train_loss = train_loss / total_samples
                avg_val_loss = val_loss / val_samples
                avg_train_accuracy = train_accuracy / total_samples
                avg_val_accuracy = val_accuracy / val_samples
                avg_grad_norm = total_norm / count_norm if count_norm > 0 else 0.0
                
                # Update learning rate scheduler
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler.step()
                
                # Check early stopping
                early_stopping(avg_val_loss)
                
                # Update monitor
                monitor.update(
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    train_accuracy=avg_train_accuracy,
                    val_accuracy=avg_val_accuracy,
                    learning_rate=optimizer.param_groups[0]['lr'],
                    gradient_norm=avg_grad_norm,
                    batch_size=batch_size,
                    num_samples=total_samples,
                    worker_id=worker_id
                )
                
                # Create plots
                if (epoch + 1) % config['training']['plot_frequency'] == 0:
                    monitor.plot()
                
                # Submit epoch end report
                report_scheduler.submit_report(worker_id, 'epoch_end', {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_acc': avg_train_accuracy,
                    'val_acc': avg_val_accuracy,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
                
                # Save checkpoint
                if (epoch + 1) % config['training']['checkpoint']['save_every'] == 0:
                    is_best = avg_val_loss < best_val_loss
                    if is_best:
                        best_val_loss = avg_val_loss
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        train_loss=avg_train_loss,
                        val_loss=avg_val_loss,
                        is_best=is_best
                    )
                
                # Clean up old checkpoints
                if (epoch + 1) % config['training']['checkpoint']['save_every'] == 0:
                    cleanup_old_checkpoints(
                        keep_last=config['training']['checkpoint']['keep_last']
                    )
                
                if early_stopping.early_stop:
                    report_scheduler.submit_report(worker_id, 'info', {
                        'message': f"Early stopping triggered at epoch {epoch + 1}"
                    })
                    break
            
            # Save final training summary
            monitor.save_summary()
        
        except Exception as e:
            report_scheduler.submit_report(worker_id, 'error', {
                'message': f"Training interrupted in worker {worker_id}: {str(e)}"
            })
            raise
    
    except Exception as e:
        report_scheduler.submit_report(worker_id, 'error', {
            'message': f"Worker {worker_id} failed: {str(e)}"
        })
        raise

def cleanup_old_checkpoints(keep_last: int) -> None:
    """Clean up old checkpoints, keeping only the specified number of most recent ones"""
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_epoch_')]
    if len(checkpoints) <= keep_last:
        return
    
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for checkpoint in checkpoints[:-keep_last]:
        os.remove(os.path.join(CHECKPOINT_DIR, checkpoint))

def main():
    """Main training function"""
    try:
        # Load configuration
        config = load_config()
        
        # Get device based on config
        device = get_device(config)
        
        # Get parameters from config
        total_epochs = config['training']['epochs']
        
        # Calculate epochs per worker
        epochs_per_worker = total_epochs // NUM_WORKERS
        remaining_epochs = total_epochs % NUM_WORKERS
        
        # Initialize report scheduler
        report_scheduler = ReportScheduler(NUM_WORKERS)
        
        # Log training configuration
        logging.info("Starting training with configuration:")
        logging.info(f"Device: {device}")
        logging.info(f"Total epochs: {total_epochs}")
        logging.info(f"Workers: {NUM_WORKERS}")
        logging.info(f"Epochs per worker: {epochs_per_worker}")
        logging.info(f"Batch size: {config['training']['batch_size']}")
        logging.info(f"Learning rate: {config['training']['learning_rate']}")
        logging.info(f"Optimizer: {config['training']['optimizer']}")
        logging.info(f"Loss function: {config['training']['loss_function']}")
        if 'scheduler' in config:
            logging.info(f"Scheduler type: {config['scheduler']['type']}")
        else:
            logging.info("Using default scheduler: CosineAnnealingWarmRestarts")
        
        # Create worker processes
        processes = []
        for i in range(NUM_WORKERS):
            # Calculate start epoch for this worker
            start_epoch = i * epochs_per_worker
            # Add remaining epochs to the last worker
            worker_epochs = epochs_per_worker + (remaining_epochs if i == NUM_WORKERS - 1 else 0)
            
            p = Process(
                target=train_worker,
                args=(i, config, device, worker_epochs, start_epoch, report_scheduler)
            )
            processes.append(p)
            p.start()
            
            # Log worker start
            logging.info(f"Started worker {i} for epochs {start_epoch}-{start_epoch + worker_epochs}")
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Stop the report scheduler
        report_scheduler.stop()
        
        logging.info("Training completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    # Set up terminal for proper progress display
    import sys
    if sys.stdout.isatty():
        # Only clear terminal if running in a terminal
        os.system('clear' if os.name == 'posix' else 'cls')
    
    # Configure logging with timestamp
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    main()

