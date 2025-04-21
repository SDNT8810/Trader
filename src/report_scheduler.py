import logging
import multiprocessing as mp
from typing import Dict, Any
from tqdm import tqdm
import sys
import os

class ReportScheduler:
    def __init__(self, num_workers: int, log_file: str = 'training.log'):
        """Initialize the report scheduler"""
        self.num_workers = num_workers
        self.manager_id = 0
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Clear terminal if running in interactive mode
        if sys.stdout.isatty():
            os.system('clear' if os.name == 'posix' else 'cls')
            print("\n" * (num_workers * 2))  # Make space for progress bars
        
        # Initialize progress bars for manager only
        self.train_bar = None
        self.val_bar = None
    
    def _create_progress_bar(self, total: int, desc: str, position: int) -> tqdm:
        """Create a progress bar with consistent settings"""
        return tqdm(
            total=total,
            desc=desc,
            position=position,
            ncols=100,
            bar_format='{l_bar}{bar:20}{r_bar}',
            mininterval=4.0,
            maxinterval=4.0,
            dynamic_ncols=False
        )
    
    def submit_report(self, worker_id: int, report_type: str, data: Dict[str, Any]):
        """Submit a report to be processed"""
        # Only process reports from manager (worker 0)
        if worker_id != self.manager_id:
            return
        
        try:
            if report_type == 'init':
                logging.info(f"Using {self.num_workers} physical cores for training")
                if data.get('device_type') == 'mps':
                    logging.info("Using MPS (Metal Performance Shaders) for M1/M2 GPU")
            
            elif report_type == 'train_start':
                if self.train_bar is not None:
                    self.train_bar.close()
                self.train_bar = self._create_progress_bar(
                    total=data['total_batches'],
                    desc=f'Epoch {data["epoch"]} (Train)',
                    position=0
                )
            
            elif report_type == 'val_start':
                if self.val_bar is not None:
                    self.val_bar.close()
                self.val_bar = self._create_progress_bar(
                    total=data['total_batches'],
                    desc=f'Epoch {data["epoch"]} (Val)',
                    position=1
                )
            
            elif report_type == 'train_update':
                if self.train_bar is not None:
                    self.train_bar.update(1)
                    self.train_bar.set_postfix(data)
            
            elif report_type == 'val_update':
                if self.val_bar is not None:
                    self.val_bar.update(1)
                    self.val_bar.set_postfix(data)
            
            elif report_type == 'epoch_end':
                logging.info(f"Epoch {data['epoch']} - "
                           f"Train Loss: {data['train_loss']:.6f}, "
                           f"Val Loss: {data['val_loss']:.6f}, "
                           f"LR: {data['learning_rate']:.6f}")
            
            elif report_type == 'error':
                logging.error(f"Worker {worker_id} - {data['message']}")
            
            elif report_type == 'info':
                logging.info(f"Worker {worker_id} - {data['message']}")
        
        except Exception as e:
            logging.error(f"Error processing report: {str(e)}")
    
    def stop(self):
        """Clean up resources"""
        if self.train_bar is not None:
            self.train_bar.close()
        if self.val_bar is not None:
            self.val_bar.close() 