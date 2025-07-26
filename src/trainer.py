"""
Training logic and trainer classes for AQI prediction models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from typing import Tuple, Dict, Any, Optional
import wandb

from .losses import (
    loss_fn_regression, loss_fn_classification, accuracy, rmse, mae,
    multi_task_loss, MetricsTracker
)


class ModelTrainer:
    """Handles training and evaluation of AQI prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        use_wandb: bool = False
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metrics trackers
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
    
    def process_batch(
        self, 
        data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        is_classification: bool = False
    ) -> Tuple[torch.Tensor, float, float, float]:
        """
        Process a single batch and return loss and metrics.
        
        Args:
            data: Tuple of (street_img, satellite_img, targets)
            is_classification: Whether this is a classification task
            
        Returns:
            Tuple of (loss, accuracy, rmse_score, mae_score)
        """
        street_img, satellite_img, targets = data
        street_img = street_img.to(self.device)
        satellite_img = satellite_img.to(self.device)
        targets = targets.to(self.device)
        
        outputs = self.model(street_img, satellite_img)
        
        if is_classification:
            loss = loss_fn_classification(outputs, targets)
            acc = accuracy(outputs, targets).item()
            rmse_score = 0.0
            mae_score = 0.0
        else:
            aqi_pred, pm_pred, gas_pred = outputs
            
            target_aqi = targets[:, 0].unsqueeze(1)  # AQI (1 value)
            target_pm = targets[:, 1:3]  # PM2.5, PM10 (2 values)
            target_gas = targets[:, 3:]  # O3, CO, SO2, NO2 (4 values)
            
            # Compute multi-task loss
            loss = multi_task_loss(
                aqi_pred, target_aqi, pm_pred, target_pm, gas_pred, target_gas
            )
            acc = 0.0  # Not applicable for regression
            
            # Compute RMSE and MAE
            rmse_score = multi_task_loss(
                aqi_pred, target_aqi, pm_pred, target_pm, gas_pred, target_gas, 
                loss_fn=rmse
            ).item()
            mae_score = multi_task_loss(
                aqi_pred, target_aqi, pm_pred, target_pm, gas_pred, target_gas, 
                loss_fn=mae
            ).item()
        
        return loss, acc, rmse_score, mae_score
    
    def train_epoch(self, data_loader, is_classification: bool = False) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")
        
        for batch_idx, data in progress_bar:
            self.optimizer.zero_grad()
            
            loss, acc, rmse_score, mae_score = self.process_batch(data, is_classification)
            
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            self.train_metrics.update(loss.item(), acc, rmse_score, mae_score)
            
            # Log to wandb if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_acc": acc,
                    "batch_rmse": rmse_score,
                    "batch_mae": mae_score,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Update progress bar
            current_metrics = self.train_metrics.compute()
            progress_bar.set_postfix({
                'Loss': f"{current_metrics['loss']:.4f}",
                'RMSE': f"{current_metrics['rmse']:.4f}",
                'MAE': f"{current_metrics['mae']:.4f}"
            })
        
        epoch_metrics = self.train_metrics.compute()
        
        # Log epoch metrics to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                "epoch_train_loss": epoch_metrics['loss'],
                "epoch_train_acc": epoch_metrics['acc'],
                "epoch_train_rmse": epoch_metrics['rmse'],
                "epoch_train_mae": epoch_metrics['mae']
            })
        
        return epoch_metrics
    
    def evaluate(self, data_loader, is_classification: bool = False) -> Dict[str, float]:
        """Evaluate the model on validation/test data."""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Evaluating")
            
            for data in progress_bar:
                loss, acc, rmse_score, mae_score = self.process_batch(data, is_classification)
                self.val_metrics.update(loss.item(), acc, rmse_score, mae_score)
                
                # Update progress bar
                current_metrics = self.val_metrics.compute()
                progress_bar.set_postfix({
                    'Val Loss': f"{current_metrics['loss']:.4f}",
                    'Val RMSE': f"{current_metrics['rmse']:.4f}",
                    'Val MAE': f"{current_metrics['mae']:.4f}"
                })
        
        val_metrics = self.val_metrics.compute()
        
        # Log validation metrics to wandb
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                "epoch_val_loss": val_metrics['loss'],
                "epoch_val_acc": val_metrics['acc'],
                "epoch_val_rmse": val_metrics['rmse'],
                "epoch_val_mae": val_metrics['mae']
            })
        
        return val_metrics
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint with additional metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint


class TrainingHistory:
    """Track training history and provide visualization."""
    
    def __init__(self):
        self.history = {
            "losses": [],
            "accuracies": [],
            "rmse_scores": [],
            "mae_scores": []
        }
    
    def update(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Update history with new epoch metrics."""
        self.history["losses"].append((train_metrics['loss'], val_metrics['loss']))
        self.history["accuracies"].append((train_metrics['acc'], val_metrics['acc']))
        self.history["rmse_scores"].append((train_metrics['rmse'], val_metrics['rmse']))
        self.history["mae_scores"].append((train_metrics['mae'], val_metrics['mae']))
    
    def get_best_epoch(self, metric: str = 'loss', split: str = 'val') -> int:
        """Get the epoch with the best metric."""
        if metric not in self.history:
            raise ValueError(f"Metric {metric} not found in history")
        
        metric_values = self.history[metric]
        split_idx = 1 if split == 'val' else 0
        
        values = [epoch_metrics[split_idx] for epoch_metrics in metric_values]
        
        if metric in ['loss']:  # Lower is better
            return np.argmin(values)
        else:  # Higher is better
            return np.argmax(values)
    
    def get_history_dict(self) -> Dict[str, Any]:
        """Get history as dictionary for plotting."""
        return self.history.copy()
