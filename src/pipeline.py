import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any
import wandb

from .config import (
    SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, IMG_SIZE,
    NUM_FROZEN_LAYERS, DROPOUT, LABELS, SATELLITE_ENCODER, STREET_ENCODER,
    ATTENTION_TYPE, SATELLITE_EXTRA_LAYER, STREET_EXTRA_LAYER
)
from .models import BaseEncoder, AQIPrediction
from .data_processing import prepare_datasets
from .trainer import ModelTrainer, TrainingHistory
from .visualization import plot_training_metrics, create_training_summary_plot


class AQITrainingPipeline:
    """Complete training pipeline for AQI prediction models."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_wandb: bool = False,
        wandb_project: str = "AQI-Detection"
    ):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration dictionary (optional, uses defaults from config.py)
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: WandB project name
        """
        self.config = self._merge_config(config)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.history = TrainingHistory()
        
    def _merge_config(self, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge user config with defaults from config.py."""
        default_config = {
            'seed': SEED,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'device': DEVICE,
            'img_size': IMG_SIZE,
            'num_frozen_layers': NUM_FROZEN_LAYERS,
            'dropout': DROPOUT,
            'labels': LABELS,
            'satellite_encoder': SATELLITE_ENCODER,
            'street_encoder': STREET_ENCODER,
            'attention_type': ATTENTION_TYPE,
            'satellite_extra_layer': SATELLITE_EXTRA_LAYER,
            'street_extra_layer': STREET_EXTRA_LAYER
        }
        
        if user_config:
            default_config.update(user_config)
            
        return default_config
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['seed'])
            torch.cuda.manual_seed_all(self.config['seed'])
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.use_wandb:
            return
            
        run_name = f"st-{self.config['street_encoder']}_sv-{self.config['satellite_encoder']}_attn-{self.config['attention_type']}"
        
        tags = [
            f"Satellite_{self.config['satellite_encoder']}_EL-{self.config['satellite_extra_layer']}",
            f"Street_{self.config['street_encoder']}_EL-{self.config['street_extra_layer']}",
            "7Bands",
            self.config['attention_type']
        ]
        
        wandb.init(
            project=self.wandb_project,
            name=run_name,
            tags=tags,
            resume='allow',
            allow_val_change=True,
            config=self.config
        )
    
    def setup_model(self) -> nn.Module:
        """Create and configure the model."""
        # Create encoders
        satellite_encoder = BaseEncoder(
            arch=self.config['satellite_encoder'],
            no_channels=7,  # 7-band satellite images
            dropout=self.config['dropout'],
            add_block=self.config['satellite_extra_layer'],
            num_frozen=self.config['num_frozen_layers']
        )
        
        street_encoder = BaseEncoder(
            arch=self.config['street_encoder'],
            no_channels=3,  # RGB street images
            dropout=self.config['dropout'],
            add_block=self.config['street_extra_layer'],
            num_frozen=self.config['num_frozen_layers']
        )
        
        # Create the combined model
        model = AQIPrediction(
            satellite_model=satellite_encoder,
            street_model=street_encoder,
            attention_type=self.config['attention_type'],
            dropout=self.config['dropout'],
            num_classes=None  # Regression task
        )
        
        self.model = model
        return model
    
    def setup_dataloaders(
        self,
        dataset_path: str,
        train_csv: pd.DataFrame,
        val_csv: pd.DataFrame,
        test_csv: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup data loaders for training, validation, and testing."""
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            train_csv, val_csv, test_csv, dataset_path, self.config['labels']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader, test_loader
    
    def setup_optimizer_and_scheduler(
        self, 
        train_loader: DataLoader
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Setup optimizer and learning rate scheduler."""
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-5
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['learning_rate'],
            steps_per_epoch=len(train_loader),
            epochs=self.config['epochs']
        )
        
        return optimizer, scheduler
    
    def train(
        self,
        dataset_path: str,
        train_csv: pd.DataFrame,
        val_csv: pd.DataFrame,
        test_csv: pd.DataFrame,
        save_dir: str = "checkpoints"
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Run the complete training pipeline.
        
        Args:
            dataset_path: Path to dataset directory
            train_csv: Training data DataFrame
            val_csv: Validation data DataFrame
            test_csv: Test data DataFrame
            save_dir: Directory to save checkpoints
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        print("🚀 Starting AQI Training Pipeline")
        print("=" * 60)
        
        # Setup WandB
        self._setup_wandb()
        
        # Setup model
        print("📦 Setting up model...")
        model = self.setup_model()
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Setup data loaders
        print("📊 Setting up data loaders...")
        train_loader, val_loader, test_loader = self.setup_dataloaders(
            dataset_path, train_csv, val_csv, test_csv
        )
        print(f"✅ Data loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_and_scheduler(train_loader)
        
        # Setup trainer
        self.trainer = ModelTrainer(
            model=model,
            device=self.config['device'],
            optimizer=optimizer,
            scheduler=scheduler,
            use_wandb=self.use_wandb
        )
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, f"best_model_{self.config['street_encoder']}_{self.config['satellite_encoder']}.pth")
        
        # Training loop
        print(f"🎯 Starting training for {self.config['epochs']} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n📈 Epoch {epoch}/{self.config['epochs']}")
            print("-" * 40)
            
            # Train for one epoch
            train_metrics = self.trainer.train_epoch(train_loader, is_classification=False)
            
            # Evaluate on validation set
            val_metrics = self.trainer.evaluate(val_loader, is_classification=False)
            
            # Update history
            self.history.update(train_metrics, val_metrics)
            
            # Print epoch summary
            print(f"Train - Loss: {train_metrics['loss']:.4f}, RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.trainer.save_checkpoint(best_model_path, epoch, val_metrics)
                print(f"💾 New best model saved (Val Loss: {best_val_loss:.4f})")
        
        # Load best model for testing
        print("\n🧪 Evaluating on test set...")
        self.trainer.load_checkpoint(best_model_path)
        test_metrics = self.trainer.evaluate(test_loader, is_classification=False)
        
        print(f"Test Results - Loss: {test_metrics['loss']:.4f}, RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
        
        # Log final test metrics to WandB
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                "final_test_loss": test_metrics['loss'],
                "final_test_rmse": test_metrics['rmse'],
                "final_test_mae": test_metrics['mae']
            })
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        history_dict = self.history.get_history_dict()
        
        # Plot training metrics
        plot_training_metrics(
            history_dict,
            model_name=f"{self.config['street_encoder']} + {self.config['satellite_encoder']}",
            save_path=os.path.join(save_dir, "training_metrics.png")
        )
        
        # Create comprehensive summary
        create_training_summary_plot(
            history_dict,
            self.config,
            save_path=os.path.join(save_dir, "training_summary.png")
        )
        
        # Finish WandB run
        if self.use_wandb and wandb.run is not None:
            wandb.finish()
        
        print("✅ Training completed successfully!")
        
        return model, history_dict
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        save_predictions: bool = True,
        save_dir: str = "results"
    ) -> Dict[str, float]:
        """
        Evaluate a trained model and optionally save predictions.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            save_predictions: Whether to save predictions
            save_dir: Directory to save results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Setup trainer for evaluation
        trainer = ModelTrainer(
            model=model,
            device=self.config['device'],
            optimizer=None,  # Not needed for evaluation
            use_wandb=False
        )
        
        # Evaluate
        test_metrics = trainer.evaluate(test_loader, is_classification=False)
        
        if save_predictions:
            # TODO: Implement prediction saving functionality
            os.makedirs(save_dir, exist_ok=True)
            print(f"💾 Results saved to {save_dir}")
        
        return test_metrics


def quick_train(
    dataset_path: str,
    train_csv: pd.DataFrame,
    val_csv: pd.DataFrame,
    test_csv: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    use_wandb: bool = False
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Quick training function for simple use cases.
    
    Args:
        dataset_path: Path to dataset directory
        train_csv: Training data DataFrame  
        val_csv: Validation data DataFrame
        test_csv: Test data DataFrame
        config: Optional configuration overrides
        use_wandb: Whether to use WandB logging
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    pipeline = AQITrainingPipeline(config=config, use_wandb=use_wandb)
    return pipeline.train(dataset_path, train_csv, val_csv, test_csv)
