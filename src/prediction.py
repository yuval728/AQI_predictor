"""
AQI prediction utilities with preprocessing and model inference.
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
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
                
                # Create encoders
                street_encoder = BaseEncoder(
                    arch=config['street_encoder'],
                    no_channels=3,
                    dropout=config['dropout'],
                    add_block=config['extra_layer'],
                    num_frozen=config['frozen_layers']
                )
                
                satellite_encoder = None
                if config['satellite_encoder']:
                    satellite_encoder = BaseEncoder(
                        arch=config['satellite_encoder'],
                        no_channels=config['satellite_channels'],
                        dropout=config['dropout'],
                        add_block=config['extra_layer'],
                        num_frozen=config['frozen_layers']
                    )
                
                # Create model
                model = AQIPrediction(
                    satellite_model=satellite_encoder,
                    street_model=street_encoder,
                    attention_type=config['attention_type'],
                    dropout=config['dropout'],
                    num_classes=None
                )
                
                # Load weights
                model.load_state_dict(torch.load(config['path'], map_location=self.device))
                model.to(self.device)
                model.eval()
                models.append(model)
                
                print(f"✓ {model_name} loaded successfully")
                
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        
        return models
    
    def preprocess_street_image(self, image) -> torch.Tensor:
        """Preprocess street view image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEANS, std=IMAGENET_STDS)
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def preprocess_satellite_image(self, image, is_seven_band: bool = False) -> torch.Tensor:
        """Preprocess satellite image."""
        if is_seven_band and isinstance(image, np.ndarray):
            # Handle 7-band satellite images
            tensor = torch.from_numpy(image).float()
            
            # Ensure proper shape: (channels, height, width)
            if tensor.ndim == 3 and tensor.shape[2] <= 7:
                tensor = tensor.permute(2, 0, 1)
            
            # Resize to 224x224
            tensor = F.interpolate(
                tensor.unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Normalize each channel
            for c in range(min(tensor.shape[0], len(SATELLITE_MEANS))):
                tensor[c] = (tensor[c] - SATELLITE_MEANS[c]) / SATELLITE_STDS[c]
            
            return tensor.unsqueeze(0).to(self.device)
        else:
            # Handle RGB satellite images
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8'))
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEANS, std=IMAGENET_STDS)
            ])
            
            return transform(image).unsqueeze(0).to(self.device)
    
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
        # Preprocess images
        street_tensor = self.preprocess_street_image(street_image)
        
        satellite_tensor = None
        satellite_7_tensor = None
        
        if satellite_image is not None:
            satellite_tensor = self.preprocess_satellite_image(satellite_image, is_seven_band=False)
        
        if satellite_7_bands is not None:
            satellite_7_tensor = self.preprocess_satellite_image(satellite_7_bands, is_seven_band=True)
        
        # Make predictions
        predictions = []
        model_names = list(MODEL_CONFIG.keys())
        
        for i, (model, model_name) in enumerate(zip(self.models, model_names)):
            try:
                with torch.no_grad():
                    config = MODEL_CONFIG[model_name]
                    
                    # Choose appropriate satellite input
                    if config['satellite_channels'] == 7 and satellite_7_tensor is not None:
                        sat_input = satellite_7_tensor
                    elif config['satellite_channels'] == 3 and satellite_tensor is not None:
                        sat_input = satellite_tensor
                    else:
                        sat_input = None
                    
                    # Forward pass
                    if sat_input is not None:
                        aqi, pm, gas = model(street_tensor, sat_input)
                    else:
                        aqi, pm, gas = model(street_tensor)
                    
                    # Extract predictions
                    prediction = {
                        "AQI": float(aqi.item()),
                        "PM2.5": float(pm[0, 0].item()),
                        "PM10": float(pm[0, 1].item()),
                        "O3": float(gas[0, 0].item()),
                        "CO": float(gas[0, 1].item()),
                        "SO2": float(gas[0, 2].item()),
                        "NO2": float(gas[0, 3].item())
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
    predictor = get_predictor()
    return predictor.predict(captured_image, satellite_image, satellite_7_bands)
