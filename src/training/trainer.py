import torch
import numpy as np
from tqdm.auto import tqdm
import wandb
from .loss_functions import process_batch


def train_fn(model, optimizer, scheduler, data_loader, device, is_classification=False):
    """Training function"""
    model.train()
    total_loss, total_acc, total_rmse, total_mae = 0, 0, 0, 0

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):
        optimizer.zero_grad()

        loss, acc, rmse_score, mae_score = process_batch(data, model, device, is_classification)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Aggregate metrics
        total_loss += loss.item()
        total_acc += acc
        total_rmse += rmse_score
        total_mae += mae_score
        
        # Log batch metrics to WandB if available
        if wandb.run is not None:
            wandb.log({
                "batch_loss": loss.item(),
                "batch_acc": acc,
                "batch_rmse": rmse_score,
                "batch_mae": mae_score
            })

        print(f"Batch: {i+1}/{len(data_loader)}, Loss: {total_loss / (i+1):.4f}, RMSE: {total_rmse / (i+1):.4f}, MAE: {total_mae / (i+1):.4f}", end='\r')

    print()  # Ensure proper formatting after tqdm

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = total_acc / len(data_loader)
    epoch_rmse = total_rmse / len(data_loader)
    epoch_mae = total_mae / len(data_loader)

    # Log epoch-wise metrics to WandB
    if wandb.run is not None:
        wandb.log({
            "epoch_loss": epoch_loss,
            "epoch_acc": epoch_acc,
            "epoch_rmse": epoch_rmse,
            "epoch_mae": epoch_mae
        })

    return epoch_loss, epoch_acc, epoch_rmse, epoch_mae


def eval_fn(model, data_loader, device, is_classification=False):
    """Evaluation function"""
    model.eval()
    total_loss, total_acc, total_rmse, total_mae = 0, 0, 0, 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating"):
            loss, acc, rmse_score, mae_score = process_batch(data, model, device, is_classification)

            total_loss += loss.item()
            total_acc += acc
            total_rmse += rmse_score
            total_mae += mae_score

    val_loss = total_loss / len(data_loader)
    val_acc = total_acc / len(data_loader)
    val_rmse = total_rmse / len(data_loader)
    val_mae = total_mae / len(data_loader)

    # Log validation metrics to WandB
    if wandb.run is not None:
        wandb.log({
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_rmse": val_rmse,
            "val_mae": val_mae
        })

    return val_loss, val_acc, val_rmse, val_mae


def train(model, optimizer, scheduler, train_loader, val_loader, test_loader, device, epochs=10, best_model_path='best_model.pth'):
    """Main training loop"""
    model.to(device)
    history = {"losses": [], "accuracies": [], "rmse_scores": [], "mae_scores": []}

    best_loss = np.inf

    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_metrics = train_fn(model, optimizer, scheduler, train_loader, device, is_classification=False)
        val_metrics = eval_fn(model, val_loader, device, is_classification=False)

        history["losses"].append((train_metrics[0], val_metrics[0]))
        history["accuracies"].append((train_metrics[1], val_metrics[1]))
        history["rmse_scores"].append((train_metrics[2], val_metrics[2]))
        history["mae_scores"].append((train_metrics[3], val_metrics[3]))

        print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_metrics[0]:.4f}, Train Acc: {train_metrics[1]:.4f}, Train RMSE: {train_metrics[2]:.4f}, Train MAE: {train_metrics[3]:.4f}, Val Loss: {val_metrics[0]:.4f}, Val Acc: {val_metrics[1]:.4f}, Val RMSE: {val_metrics[2]:.4f}, Val MAE: {val_metrics[3]:.4f}")

        # Save best model
        if val_metrics[0] < best_loss:
            best_loss = val_metrics[0]
            torch.save(model.state_dict(), best_model_path)

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_metrics = eval_fn(model, test_loader, device, is_classification=False)
    if wandb.run is not None:
        wandb.log({
            "test_loss": test_metrics[0],
            "test_acc": test_metrics[1],
            "test_rmse": test_metrics[2],
            "test_mae": test_metrics[3]
        })
    print(f"Test Loss: {test_metrics[0]:.4f}, Test Acc: {test_metrics[1]:.4f}, Test RMSE: {test_metrics[2]:.4f}, Test MAE: {test_metrics[3]:.4f}")

    return history
