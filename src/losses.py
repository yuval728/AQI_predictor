"""
Loss functions and metrics for AQI prediction models.
"""
import torch
import torch.nn as nn
from typing import Optional, List


def loss_fn_regression(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard regression loss function using MSE."""
    return nn.MSELoss()(outputs, targets)


def loss_fn_classification(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard classification loss function using CrossEntropy."""
    return nn.CrossEntropyLoss()(outputs, targets)


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate classification accuracy."""
    return (outputs.argmax(1) == targets).float().mean()


def rmse(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate Root Mean Square Error."""
    return torch.sqrt(nn.MSELoss()(outputs, targets))


def mae(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate Mean Absolute Error."""
    return nn.L1Loss()(outputs, targets)


def multi_task_loss(
    aqi_pred: torch.Tensor, 
    aqi_true: torch.Tensor, 
    pm_pred: torch.Tensor, 
    pm_true: torch.Tensor, 
    gas_pred: torch.Tensor, 
    gas_true: torch.Tensor, 
    loss_fn: callable = loss_fn_regression, 
    weights: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Multi-task loss function for AQI, PM, and gas predictions.
    
    Args:
        aqi_pred: Predicted AQI values
        aqi_true: Ground truth AQI values
        pm_pred: Predicted PM values (PM2.5, PM10)
        pm_true: Ground truth PM values
        gas_pred: Predicted gas values (O3, CO, SO2, NO2)
        gas_true: Ground truth gas values
        loss_fn: Loss function to use (default: MSE)
        weights: Task-specific weights [AQI, PM, Gas]
    
    Returns:
        Weighted sum of task-specific losses
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0]  # Equal weights for AQI, PM, and Gases

    loss_aqi = loss_fn(aqi_pred, aqi_true)
    loss_pm = loss_fn(pm_pred, pm_true)
    loss_gas = loss_fn(gas_pred, gas_true)
    
    total_loss = weights[0] * loss_aqi + weights[1] * loss_pm + weights[2] * loss_gas
    return total_loss


def dynamic_task_weights(y_aqi: torch.Tensor, y_pm: torch.Tensor, y_gas: torch.Tensor) -> List[float]:
    """
    Compute dynamic task weights based on inverse variance.
    Higher variance leads to lower weight.
    
    Args:
        y_aqi: AQI targets
        y_pm: PM targets
        y_gas: Gas targets
    
    Returns:
        List of normalized weights [aqi_weight, pm_weight, gas_weight]
    """
    var_aqi = torch.var(y_aqi)
    var_pm = torch.var(y_pm)
    var_gas = torch.var(y_gas)

    weights = 1 / (torch.tensor([var_aqi, var_pm, var_gas]) + 1e-6)  # Avoid division by zero
    weights /= torch.sum(weights)  # Normalize weights to sum to 1
    return weights.tolist()


class MetricsTracker:
    """Track and compute metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.total_loss = 0.0
        self.total_acc = 0.0
        self.total_rmse = 0.0
        self.total_mae = 0.0
        self.count = 0
    
    def update(self, loss: float, acc: float, rmse: float, mae: float):
        """Update metrics with new batch results."""
        self.total_loss += loss
        self.total_acc += acc
        self.total_rmse += rmse
        self.total_mae += mae
        self.count += 1
    
    def compute(self) -> dict:
        """Compute average metrics."""
        if self.count == 0:
            return {"loss": 0.0, "acc": 0.0, "rmse": 0.0, "mae": 0.0}
        
        return {
            "loss": self.total_loss / self.count,
            "acc": self.total_acc / self.count,
            "rmse": self.total_rmse / self.count,
            "mae": self.total_mae / self.count
        }
