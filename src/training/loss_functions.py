import torch
import torch.nn as nn


def loss_fn_regression(outputs, targets):
    """
    Loss function for regression
    """
    return nn.MSELoss()(outputs, targets)


def loss_fn_classification(outputs, targets):
    """
    Loss function for classification
    """
    return nn.CrossEntropyLoss()(outputs, targets)


def accuracy(outputs, targets):
    """
    Accuracy function
    """
    return (outputs.argmax(1) == targets).float().mean()


def rmse(outputs, targets):
    """
    RMSE function
    """
    return torch.sqrt(nn.MSELoss()(outputs, targets))


def mae(outputs, targets):
    """
    MAE function
    """
    return nn.L1Loss()(outputs, targets)


def multi_task_loss(aqi_pred, aqi_true, pm_pred, pm_true, gas_pred, gas_true, loss_fn=loss_fn_regression, weights=None):
    """
    Multi-task loss function for AQI, PM, and gas predictions.
    
    Args:
        aqi_pred: Predicted AQI values
        aqi_true: True AQI values
        pm_pred: Predicted PM values (PM2.5, PM10)
        pm_true: True PM values
        gas_pred: Predicted gas values (O3, CO, SO2, NO2)
        gas_true: True gas values
        loss_fn: Loss function to use (default: MSE)
        weights: Weights for different tasks (default: equal weights)
        
    Returns:
        Weighted sum of losses
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0]  # Default equal weights for AQI, PM, and Gases

    loss_aqi = loss_fn(aqi_pred, aqi_true)
    loss_pm = loss_fn(pm_pred, pm_true)
    loss_gas = loss_fn(gas_pred, gas_true)
    
    # Weighted sum of losses
    total_loss = weights[0] * loss_aqi + weights[1] * loss_pm + weights[2] * loss_gas
    return total_loss


def dynamic_task_weights(y_aqi, y_pm, y_gas):
    """
    Compute dynamic task weights based on variance of targets.
    Higher variance gets lower weight to balance the learning.
    """
    # Compute inverse variance (higher variance → lower weight)
    var_aqi = torch.var(y_aqi)
    var_pm = torch.var(y_pm)
    var_gas = torch.var(y_gas)

    weights = 1 / (torch.tensor([var_aqi, var_pm, var_gas]) + 1e-6)  # Avoid division by zero
    weights /= torch.sum(weights)  # Normalize weights to sum to 1
    return weights.tolist()


def process_batch(data, model, device, is_classification):
    """Processes a batch and returns outputs, loss, and metrics."""
    street_img, satellite_img, targets = data
    street_img, satellite_img, targets = street_img.to(device), satellite_img.to(device), targets.to(device)
    
    outputs = model(street_img, satellite_img)
    
    if is_classification:
        loss = loss_fn_classification(outputs, targets)
        acc = accuracy(outputs, targets).item()
        rmse_score = 0
        mae_score = 0
    else:
        aqi_pred, pm_pred, gas_pred = outputs
        
        target_aqi = targets[:, 0].unsqueeze(1)  # AQI (1 value)
        target_pm = targets[:, 1:3]  # PM2.5, PM10 (2 values)
        target_gas = targets[:, 3:]  # O3, CO, SO2, NO2 (4 values)

        # Optional: Use dynamic weights
        # weights = dynamic_task_weights(target_aqi, target_pm, target_gas)
        weights = None
        
        loss = multi_task_loss(aqi_pred, target_aqi, pm_pred, target_pm, gas_pred, target_gas, weights=weights)
        acc = 0  # Not applicable for regression

        rmse_score = multi_task_loss(aqi_pred, target_aqi, pm_pred, target_pm, gas_pred, target_gas, rmse, weights=weights).item()
        mae_score = multi_task_loss(aqi_pred, target_aqi, pm_pred, target_pm, gas_pred, target_gas, mae, weights=weights).item()

    return loss, acc, rmse_score, mae_score
