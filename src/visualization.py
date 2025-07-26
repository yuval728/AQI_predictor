"""
Visualization utilities for AQI prediction training and results.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import seaborn as sns


def plot_training_metrics(
    history: Dict[str, List[Tuple[float, float]]], 
    model_name: str = "Model Metrics",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 15)
):
    """
    Plot training metrics with improved styling.
    
    Args:
        history: Dictionary containing training history
        model_name: Name of the model for title
        save_path: Path to save the plot (optional)
        figsize: Figure size tuple
    """
    # Set style
    plt.style.use('seaborn-v0_8')
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'{model_name} - Training Metrics', fontsize=20, fontweight='bold')
    
    # Extract training and validation metrics
    train_losses = [epoch[0] for epoch in history['losses']]
    val_losses = [epoch[1] for epoch in history['losses']]
    
    train_accs = [epoch[0] for epoch in history['accuracies']]
    val_accs = [epoch[1] for epoch in history['accuracies']]
    
    train_rmse = [epoch[0] for epoch in history['rmse_scores']]
    val_rmse = [epoch[1] for epoch in history['rmse_scores']]
    
    train_mae = [epoch[0] for epoch in history['mae_scores']]
    val_mae = [epoch[1] for epoch in history['mae_scores']]
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot RMSE
    axes[1, 0].plot(epochs, train_rmse, 'b-', label='Training', linewidth=2)
    axes[1, 0].plot(epochs, val_rmse, 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_title('RMSE', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1, 1].plot(epochs, train_mae, 'b-', label='Training', linewidth=2)
    axes[1, 1].plot(epochs, val_mae, 'r-', label='Validation', linewidth=2)
    axes[1, 1].set_title('MAE', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prediction_comparison(
    true_values: np.ndarray,
    predictions: np.ndarray,
    labels: List[str],
    model_name: str = "Model Predictions",
    save_path: Optional[str] = None
):
    """
    Plot true vs predicted values for each output variable.
    
    Args:
        true_values: Ground truth values (n_samples, n_outputs)
        predictions: Predicted values (n_samples, n_outputs)
        labels: Labels for each output variable
        model_name: Name of the model
        save_path: Path to save the plot
    """
    n_outputs = len(labels)
    n_cols = min(3, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'{model_name} - True vs Predicted Values', fontsize=16, fontweight='bold')
    
    if n_outputs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, label in enumerate(labels):
        ax = axes[i] if n_outputs > 1 else axes[0]
        
        # Scatter plot
        ax.scatter(true_values[:, i], predictions[:, i], alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(true_values[:, i].min(), predictions[:, i].min())
        max_val = max(true_values[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel(f'True {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_title(f'{label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and display R²
        correlation_matrix = np.corrcoef(true_values[:, i], predictions[:, i])
        r_squared = correlation_matrix[0, 1] ** 2
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_outputs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_loss_landscape(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss curves with enhanced styling.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(epochs, train_losses, 'b-', linewidth=3, label='Training Loss', alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', linewidth=3, label='Validation Loss', alpha=0.8)
    
    # Find minimum validation loss
    min_val_loss_epoch = np.argmin(val_losses) + 1
    min_val_loss = min(val_losses)
    
    plt.axvline(x=min_val_loss_epoch, color='green', linestyle='--', alpha=0.7, 
                label=f'Best Validation (Epoch {min_val_loss_epoch})')
    plt.scatter([min_val_loss_epoch], [min_val_loss], color='green', s=100, zorder=5)
    
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.annotate(f'Min Val Loss: {min_val_loss:.4f}', 
                xy=(min_val_loss_epoch, min_val_loss), 
                xytext=(min_val_loss_epoch + len(epochs)*0.1, min_val_loss + (max(val_losses) - min_val_loss)*0.1),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None
):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights matrix
        feature_names: Names of the features
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(attention_weights, 
                xticklabels=feature_names,
                yticklabels=range(1, len(attention_weights) + 1),
                annot=True, 
                cmap='viridis',
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title('Attention Weights Visualization', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Samples', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_training_summary_plot(
    history: Dict,
    model_config: Dict,
    save_path: Optional[str] = None
):
    """
    Create a comprehensive training summary plot.
    
    Args:
        history: Training history dictionary
        model_config: Model configuration dictionary
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main metrics plot
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    # Extract metrics
    train_losses = [epoch[0] for epoch in history['losses']]
    val_losses = [epoch[1] for epoch in history['losses']]
    epochs = range(1, len(train_losses) + 1)
    
    ax_main.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax_main.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax_main.set_title('Training Progress', fontsize=16, fontweight='bold')
    ax_main.set_xlabel('Epoch')
    ax_main.set_ylabel('Loss')
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # RMSE subplot
    ax_rmse = fig.add_subplot(gs[0, 2:])
    train_rmse = [epoch[0] for epoch in history['rmse_scores']]
    val_rmse = [epoch[1] for epoch in history['rmse_scores']]
    ax_rmse.plot(epochs, train_rmse, 'b-', label='Training RMSE', linewidth=2)
    ax_rmse.plot(epochs, val_rmse, 'r-', label='Validation RMSE', linewidth=2)
    ax_rmse.set_title('RMSE Progress')
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3)
    
    # MAE subplot
    ax_mae = fig.add_subplot(gs[1, 2:])
    train_mae = [epoch[0] for epoch in history['mae_scores']]
    val_mae = [epoch[1] for epoch in history['mae_scores']]
    ax_mae.plot(epochs, train_mae, 'b-', label='Training MAE', linewidth=2)
    ax_mae.plot(epochs, val_mae, 'r-', label='Validation MAE', linewidth=2)
    ax_mae.set_title('MAE Progress')
    ax_mae.legend()
    ax_mae.grid(True, alpha=0.3)
    
    # Configuration text
    ax_config = fig.add_subplot(gs[2, :])
    ax_config.axis('off')
    
    config_text = f"""
    Model Configuration:
    • Architecture: {model_config.get('arch', 'N/A')}
    • Batch Size: {model_config.get('batch_size', 'N/A')}
    • Learning Rate: {model_config.get('learning_rate', 'N/A')}
    • Epochs: {len(epochs)}
    • Best Validation Loss: {min(val_losses):.4f} (Epoch {np.argmin(val_losses) + 1})
    • Final Training Loss: {train_losses[-1]:.4f}
    • Final Validation Loss: {val_losses[-1]:.4f}
    """
    
    ax_config.text(0.1, 0.5, config_text, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Training Summary Report', fontsize=20, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
