"""
Script to train AQI prediction models.
"""
import torch
from torch.utils.data import DataLoader
import argparse
import json
from src.training import AQIDataset, AQITrainer, create_model, prepare_data, get_transforms
from src.config import MODEL_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Train AQI Prediction Model')
    parser.add_argument('--config', type=str, default='efficientnet_b0', 
                       help='Model configuration name')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV file with data')
    parser.add_argument('--street_img_dir', type=str, required=True,
                       help='Directory containing street images')
    parser.add_argument('--satellite_img_dir', type=str, required=True,
                       help='Directory containing satellite images')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                       help='Directory to save trained models')
    parser.add_argument('--use_7_bands', action='store_true',
                       help='Use 7-band satellite images')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    if args.config in MODEL_CONFIG:
        config = MODEL_CONFIG[args.config]
    else:
        print(f"Configuration '{args.config}' not found. Available configs: {list(MODEL_CONFIG.keys())}")
        return
    
    # Prepare data
    print("Preparing data splits...")
    train_data, val_data, test_data = prepare_data(
        args.data_path, 
        args.street_img_dir, 
        args.satellite_img_dir
    )
    
    # Get transforms
    transforms = get_transforms()
    
    # Create datasets
    train_dataset = AQIDataset(
        train_data, 
        args.street_img_dir, 
        args.satellite_img_dir,
        transform_street=transforms['train_street'],
        transform_satellite=transforms['train_satellite'],
        use_7_bands=args.use_7_bands
    )
    
    val_dataset = AQIDataset(
        val_data, 
        args.street_img_dir, 
        args.satellite_img_dir,
        transform_street=transforms['val_street'],
        transform_satellite=transforms['val_satellite'],
        use_7_bands=args.use_7_bands
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    print(f"Creating model with configuration: {args.config}")
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = AQITrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate
    )
    
    # Train model
    save_path = f"{args.output_dir}/{args.config}_best_model.pth"
    print(f"Starting training... Model will be saved to {save_path}")
    
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=save_path
    )
    
    # Save training history
    history_path = f"{args.output_dir}/{args.config}_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Training completed! Best validation loss: {training_history['best_val_loss']:.6f}")
    print(f"Model saved to: {save_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
