import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(losses, accuracies, rmse_scores, mae_scores, model_name="Model Metrics"):
    """
    Plot training metrics with a title and save functionality.
    
    Args:
        losses: List of tuples (train_loss, val_loss) for each epoch
        accuracies: List of tuples (train_acc, val_acc) for each epoch
        rmse_scores: List of tuples (train_rmse, val_rmse) for each epoch
        mae_scores: List of tuples (train_mae, val_mae) for each epoch
        model_name: Title for the plot
    """
    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(model_name, fontsize=20, fontweight="bold")

    # Plot losses
    train_losses, val_losses = zip(*losses)
    ax[0, 0].plot(train_losses, label="Train", color='blue')
    ax[0, 0].plot(val_losses, label="Val", color='orange')
    ax[0, 0].set_title("Loss")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # Plot accuracies
    train_accs, val_accs = zip(*accuracies)
    ax[0, 1].plot(train_accs, label="Train", color='blue')
    ax[0, 1].plot(val_accs, label="Val", color='orange')
    ax[0, 1].set_title("Accuracy")
    ax[0, 1].set_xlabel("Epoch")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # Plot RMSE scores
    train_rmses, val_rmses = zip(*rmse_scores)
    ax[1, 0].plot(train_rmses, label="Train", color='blue')
    ax[1, 0].plot(val_rmses, label="Val", color='orange')
    ax[1, 0].set_title("RMSE")
    ax[1, 0].set_xlabel("Epoch")
    ax[1, 0].set_ylabel("RMSE")
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Plot MAE scores
    train_maes, val_maes = zip(*mae_scores)
    ax[1, 1].plot(train_maes, label="Train", color='blue')
    ax[1, 1].plot(val_maes, label="Val", color='orange')
    ax[1, 1].set_title("MAE")
    ax[1, 1].set_xlabel("Epoch")
    ax[1, 1].set_ylabel("MAE")
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()
    
    return fig


def plot_training_curves(history, save_path=None):
    """
    Plot training curves from history dictionary.
    
    Args:
        history: Dictionary with keys 'losses', 'accuracies', 'rmse_scores', 'mae_scores'
        save_path: Optional path to save the plot
    """
    fig = plot_metrics(
        history['losses'],
        history['accuracies'],
        history['rmse_scores'],
        history['mae_scores']
    )
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    return fig


def plot_predictions_vs_actual(predictions, actuals, labels=None, title="Predictions vs Actual"):
    """
    Plot predictions vs actual values for multiple outputs.
    
    Args:
        predictions: List of prediction arrays for each output
        actuals: List of actual value arrays for each output
        labels: List of labels for each output
        title: Plot title
    """
    if labels is None:
        labels = [f"Output {i+1}" for i in range(len(predictions))]
    
    n_outputs = len(predictions)
    cols = min(3, n_outputs)
    rows = (n_outputs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_outputs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    for i, (pred, actual, label) in enumerate(zip(predictions, actuals, labels)):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Scatter plot
        ax.scatter(actual, pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(np.min(actual), np.min(pred))
        max_val = max(np.max(actual), np.max(pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel(f'Actual {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_title(f'{label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_outputs, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_attention_weights(attention_weights, save_path=None):
    """
    Visualize attention weights if available.
    
    Args:
        attention_weights: Attention weight tensor
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to numpy if it's a tensor
    if hasattr(attention_weights, 'detach'):
        weights = attention_weights.detach().cpu().numpy()
    else:
        weights = attention_weights
    
    # Plot as heatmap
    im = ax.imshow(weights, cmap='Blues', aspect='auto')
    ax.set_title('Attention Weights Visualization')
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Batch Sample')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention plot saved to {save_path}")
    
    plt.show()
    return fig
