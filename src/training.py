"""
Training utilities for AQI prediction models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from typing import Tuple
from src.models import BaseEncoder, AQIPrediction
from src.config import IMAGENET_MEANS, IMAGENET_STDS
import torchvision.transforms as transforms


class AQIDataset(Dataset):
    """Dataset class for AQI prediction."""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 street_img_dir: str,
                 satellite_img_dir: str,
                 transform_street=None,
                 transform_satellite=None,
                 use_7_bands: bool = False):
        
        self.data = data
        self.street_img_dir = street_img_dir
        self.satellite_img_dir = satellite_img_dir
        self.transform_street = transform_street
        self.transform_satellite = transform_satellite
        self.use_7_bands = use_7_bands
        
        # Define labels
        self.labels = ['AQI', 'PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load street image
        street_img_path = os.path.join(self.street_img_dir, row['Filename'])
        street_img = Image.open(street_img_path).convert('RGB')
        
        if self.transform_street:
            street_img = self.transform_street(street_img)
        
        # Load satellite image
        satellite_img_path = os.path.join(self.satellite_img_dir, f"{row['Normalized_Filename']}.jpg")
        
        if self.use_7_bands:
            # Load 7-band satellite data (assuming saved as .npy)
            sat_path = satellite_img_path.replace('.jpg', '.npy')
            if os.path.exists(sat_path):
                satellite_img = np.load(sat_path)
                satellite_img = torch.from_numpy(satellite_img).float()
            else:
                # Fallback to RGB if 7-band not available
                satellite_img = Image.open(satellite_img_path).convert('RGB')
                if self.transform_satellite:
                    satellite_img = self.transform_satellite(satellite_img)
        else:
            satellite_img = Image.open(satellite_img_path).convert('RGB')
            if self.transform_satellite:
                satellite_img = self.transform_satellite(satellite_img)
        
        # Get labels
        targets = {
            'aqi': torch.tensor(row['AQI'], dtype=torch.float32),
            'pm': torch.tensor([row['PM2.5'], row['PM10']], dtype=torch.float32),
            'gas': torch.tensor([row['O3'], row['CO'], row['SO2'], row['NO2']], dtype=torch.float32)
        }
        
        return street_img, satellite_img, targets


def get_transforms(image_size: int = 224):
    """Get data transforms for training and validation."""
    
    # Street image transforms
    train_transform_street = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEANS, std=IMAGENET_STDS)
    ])
    
    val_transform_street = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEANS, std=IMAGENET_STDS)
    ])
    
    # Satellite image transforms
    train_transform_satellite = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEANS, std=IMAGENET_STDS)
    ])
    
    val_transform_satellite = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEANS, std=IMAGENET_STDS)
    ])
    
    return {
        'train_street': train_transform_street,
        'val_street': val_transform_street,
        'train_satellite': train_transform_satellite,
        'val_satellite': val_transform_satellite
    }


class AQITrainer:
    """Trainer class for AQI prediction models."""
    
    def __init__(self, 
                 model: AQIPrediction,
                 device: torch.device,
                 learning_rate: float = 2e-2,
                 weight_decay: float = 1e-4):
        
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for street_imgs, satellite_imgs, targets in train_loader:
            street_imgs = street_imgs.to(self.device)
            satellite_imgs = satellite_imgs.to(self.device)
            
            aqi_targets = targets['aqi'].to(self.device).unsqueeze(1)
            pm_targets = targets['pm'].to(self.device)
            gas_targets = targets['gas'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            aqi_pred, pm_pred, gas_pred = self.model(street_imgs, satellite_imgs)
            
            # Calculate losses
            aqi_loss = self.criterion(aqi_pred, aqi_targets)
            pm_loss = self.criterion(pm_pred, pm_targets)
            gas_loss = self.criterion(gas_pred, gas_targets)
            
            total_loss_batch = aqi_loss + pm_loss + gas_loss
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for street_imgs, satellite_imgs, targets in val_loader:
                street_imgs = street_imgs.to(self.device)
                satellite_imgs = satellite_imgs.to(self.device)
                
                aqi_targets = targets['aqi'].to(self.device).unsqueeze(1)
                pm_targets = targets['pm'].to(self.device)
                gas_targets = targets['gas'].to(self.device)
                
                # Forward pass
                aqi_pred, pm_pred, gas_pred = self.model(street_imgs, satellite_imgs)
                
                # Calculate losses
                aqi_loss = self.criterion(aqi_pred, aqi_targets)
                pm_loss = self.criterion(pm_pred, pm_targets)
                gas_loss = self.criterion(gas_pred, gas_targets)
                
                total_loss_batch = aqi_loss + pm_loss + gas_loss
                total_loss += total_loss_batch.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              epochs: int,
              save_path: str = 'best_model.pth',
              patience: int = 10) -> dict:
        """Train the model with early stopping."""
        
        print(f"Starting training for {epochs} epochs...")
        early_stop_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 50)
        
        # Load best model
        self.model.load_state_dict(torch.load(save_path))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': epoch + 1
        }


def create_model(config: dict) -> AQIPrediction:
    """Create a model from configuration."""
    
    # Create street encoder
    street_encoder = BaseEncoder(
        arch=config['street_encoder'],
        no_channels=3,
        dropout=config.get('dropout', 0.5),
        add_block=config.get('extra_layer', False),
        num_frozen=config.get('frozen_layers', 0)
    )
    
    # Create satellite encoder (if specified)
    satellite_encoder = None
    if config.get('satellite_encoder'):
        satellite_encoder = BaseEncoder(
            arch=config['satellite_encoder'],
            no_channels=config.get('satellite_channels', 3),
            dropout=config.get('dropout', 0.5),
            add_block=config.get('extra_layer', False),
            num_frozen=config.get('frozen_layers', 0)
        )
    
    # Create model
    model = AQIPrediction(
        satellite_model=satellite_encoder,
        street_model=street_encoder,
        attention_type=config.get('attention_type'),
        dropout=config.get('dropout', 0.5),
        num_classes=None
    )
    
    return model


def prepare_data(csv_path: str, 
                 street_img_dir: str, 
                 satellite_img_dir: str,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare train/val/test splits from data."""
    
    # Load data
    data = pd.read_csv(csv_path)
    
    # Remove rows with missing images
    valid_indices = []
    for idx, row in data.iterrows():
        street_path = os.path.join(street_img_dir, row['Filename'])
        sat_path = os.path.join(satellite_img_dir, f"{row['Normalized_Filename']}.jpg")
        
        if os.path.exists(street_path) and os.path.exists(sat_path):
            valid_indices.append(idx)
    
    data = data.loc[valid_indices].reset_index(drop=True)
    print(f"Found {len(data)} samples with valid images")
    
    # Split data
    train_data, temp_data = train_test_split(
        data, test_size=(test_size + val_size), random_state=random_state, 
        stratify=data['AQI_Class'] if 'AQI_Class' in data.columns else None
    )
    
    val_data, test_data = train_test_split(
        temp_data, test_size=(test_size / (test_size + val_size)), random_state=random_state,
        stratify=temp_data['AQI_Class'] if 'AQI_Class' in temp_data.columns else None
    )
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data
