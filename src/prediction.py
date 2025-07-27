"""
AQI prediction utilities with preprocessing and model inference.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
import os
from src.models import BaseEncoder, AQIPrediction
from src.config import MODEL_CONFIG, IMAGENET_MEANS, IMAGENET_STDS, SATELLITE_MEANS, SATELLITE_STDS


class AQIPredictor:
    """Handle AQI prediction with multiple models."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.models = self._load_models()
    
    def _load_models(self) -> List[torch.nn.Module]:
        """Load all configured models."""
        models = []
        
        for model_name, config in MODEL_CONFIG.items():
            try:
                print(f"Loading {model_name}...")
                
                # Check if model file exists
                if not os.path.exists(config['path']):
                    print(f"✗ Model file not found: {config['path']}")
                    continue
                
                # Create encoders
                street_encoder = BaseEncoder(
                    arch=config['street_encoder'],
                    no_channels=3,
                    dropout=config.get('dropout', 0.0),
                    add_block=config.get('extra_layer', False),
                    num_frozen=config.get('frozen_layers', 0)
                )
                
                satellite_encoder = None
                if config.get('satellite_encoder'):
                    satellite_encoder = BaseEncoder(
                        arch=config['satellite_encoder'],
                        no_channels=config.get('satellite_channels', 3),
                        dropout=config.get('dropout', 0.0),
                        add_block=config.get('extra_layer', False),
                        num_frozen=config.get('frozen_layers', 0)
                    )
                
                # Create model
                model = AQIPrediction(
                    satellite_model=satellite_encoder,
                    street_model=street_encoder,
                    attention_type=config.get('attention_type', 'none'),
                    dropout=config.get('dropout', 0.0),
                    num_classes=None
                )
                
                # Load weights with error handling
                try:
                    checkpoint = torch.load(config['path'], map_location=self.device, weights_only=True)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                except Exception as load_error:
                    print(f"✗ Failed to load weights for {model_name}: {load_error}")
                    continue
                
                model.to(self.device)
                model.eval()
                models.append(model)
                
                print(f"✓ {model_name} loaded successfully")
                
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        
        if not models:
            print("Warning: No models were loaded successfully!")
        
        return models
    
    def preprocess_street_image(self, image) -> torch.Tensor:
        """Preprocess street view image."""
        try:
            if isinstance(image, np.ndarray):
                # Ensure proper data type and range
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:  # Normalized image
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEANS, std=IMAGENET_STDS)
            ])
            
            return transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to preprocess street image: {e}")
    
    def preprocess_satellite_image(self, image, is_seven_band: bool = False) -> torch.Tensor:
        """Preprocess satellite image."""
        try:
            if is_seven_band and isinstance(image, np.ndarray):
                # Handle 7-band satellite images
                if image.ndim != 3:
                    raise ValueError(f"Expected 3D array for 7-band image, got {image.ndim}D")
                
                tensor = torch.from_numpy(image).float()
                
                # Ensure proper shape: (channels, height, width)
                if tensor.shape[2] <= 7:  # (H, W, C) format
                    tensor = tensor.permute(2, 0, 1)
                elif tensor.shape[0] > 7:  # Invalid format
                    raise ValueError(f"Invalid 7-band image shape: {tensor.shape}")
                
                # Pad or truncate to exactly 7 channels
                if tensor.shape[0] < 7:
                    padding = torch.zeros(7 - tensor.shape[0], tensor.shape[1], tensor.shape[2])
                    tensor = torch.cat([tensor, padding], dim=0)
                elif tensor.shape[0] > 7:
                    tensor = tensor[:7]
                
                # Resize to 224x224
                tensor = F.interpolate(
                    tensor.unsqueeze(0), 
                    size=(224, 224), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
                # Normalize each channel with proper bounds checking
                means = SATELLITE_MEANS[:7] if len(SATELLITE_MEANS) >= 7 else SATELLITE_MEANS + [0.0] * (7 - len(SATELLITE_MEANS))
                stds = SATELLITE_STDS[:7] if len(SATELLITE_STDS) >= 7 else SATELLITE_STDS + [1.0] * (7 - len(SATELLITE_STDS))
                
                for c in range(7):
                    tensor[c] = (tensor[c] - means[c]) / max(stds[c], 1e-8)  # Avoid division by zero
                
                return tensor.unsqueeze(0).to(self.device)
            else:
                # Handle RGB satellite images
                if isinstance(image, np.ndarray):
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    image = Image.fromarray(image)
                elif not isinstance(image, Image.Image):
                    raise ValueError(f"Unsupported image type: {type(image)}")
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEANS, std=IMAGENET_STDS)
                ])
                
                return transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to preprocess satellite image: {e}")
    
    def predict(self, street_image, satellite_image=None, satellite_7_bands=None) -> List[Dict[str, Any]]:
        """
        Predict AQI and air quality metrics.
        
        Args:
            street_image: Street view image (PIL Image or numpy array)
            satellite_image: RGB satellite image (PIL Image or numpy array)
            satellite_7_bands: 7-band satellite image (numpy array)
        
        Returns:
            List of predictions from all models
        """
        if not self.models:
            return [{"error": "No models available for prediction"}]
        
        try:
            # Preprocess images
            street_tensor = self.preprocess_street_image(street_image)
            
            satellite_tensor = None
            satellite_7_tensor = None
            
            if satellite_image is not None:
                satellite_tensor = self.preprocess_satellite_image(satellite_image, is_seven_band=False)
            
            if satellite_7_bands is not None:
                satellite_7_tensor = self.preprocess_satellite_image(satellite_7_bands, is_seven_band=True)
            
        except Exception as e:
            return [{"error": f"Preprocessing failed: {e}"}]
        
        # Make predictions
        predictions = []
        model_names = list(MODEL_CONFIG.keys())
        
        for i, (model, model_name) in enumerate(zip(self.models, model_names)):
            try:
                with torch.no_grad():
                    config = MODEL_CONFIG[model_name]
                    
                    # Choose appropriate satellite input
                    sat_input = None
                    if config.get('satellite_channels') == 7 and satellite_7_tensor is not None:
                        sat_input = satellite_7_tensor
                    elif config.get('satellite_channels', 3) == 3 and satellite_tensor is not None:
                        sat_input = satellite_tensor
                    
                    # Forward pass
                    if sat_input is not None:
                        outputs = model(street_tensor, sat_input)
                    else:
                        outputs = model(street_tensor)
                    
                    # Handle different output formats
                    if isinstance(outputs, tuple) and len(outputs) == 3:
                        aqi, pm, gas = outputs
                    elif isinstance(outputs, torch.Tensor):
                        # Single output tensor - split based on model architecture
                        if outputs.shape[1] >= 7:  # AQI + 6 pollutants
                            aqi = outputs[:, 0:1]
                            pm = outputs[:, 1:3]  # PM2.5, PM10
                            gas = outputs[:, 3:7]  # O3, CO, SO2, NO2
                        else:
                            raise ValueError(f"Unexpected output shape: {outputs.shape}")
                    else:
                        raise ValueError(f"Unexpected model output format: {type(outputs)}")
                    
                    # Extract predictions with bounds checking
                    prediction = {
                        "AQI": max(0.0, float(aqi.squeeze().item())),
                        "PM2.5": max(0.0, float(pm[0, 0].item()) if pm.shape[1] > 0 else 0.0),
                        "PM10": max(0.0, float(pm[0, 1].item()) if pm.shape[1] > 1 else 0.0),
                        "O3": max(0.0, float(gas[0, 0].item()) if gas.shape[1] > 0 else 0.0),
                        "CO": max(0.0, float(gas[0, 1].item()) if gas.shape[1] > 1 else 0.0),
                        "SO2": max(0.0, float(gas[0, 2].item()) if gas.shape[1] > 2 else 0.0),
                        "NO2": max(0.0, float(gas[0, 3].item()) if gas.shape[1] > 3 else 0.0)
                    }
                    
                    predictions.append({
                        "model": model_name,
                        "predictions": prediction
                    })
                    
            except Exception as e:
                print(f"Error in prediction with {model_name}: {e}")
                predictions.append({
                    "model": model_name,
                    "predictions": None,
                    "error": str(e)
                })
        
        return predictions


# Global predictor instance
_predictor = None

def get_predictor() -> AQIPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = AQIPredictor()
    return _predictor


def predict_aqi(captured_image: Image.Image, 
                satellite_image: Optional[Image.Image] = None, 
                satellite_7_bands: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """
    Predict AQI using street and satellite images (backward compatibility).
    
    Args:
        captured_image: Street view image
        satellite_image: RGB satellite image
        satellite_7_bands: 7-band satellite image array
    
    Returns:
        List of predictions from all models
    """
    try:
        predictor = get_predictor()
        return predictor.predict(captured_image, satellite_image, satellite_7_bands)
    except Exception as e:
        return [{"error": f"Prediction failed: {e}"}]
