import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.resnet_aqi import AQIPrediction
from utils.resnet_base import BaseEncoder

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA Available:", torch.cuda.is_available())
print("Using device:", device)

SATELLITE_ENCODER = 'resnet18'
STREET_ENCODER = 'resnet18'
DROPOUT = 0.5
EXTRA_LAYER = True
NUM_FROZEN_LAYERS = 0

satellite_encoder = BaseEncoder(arch=SATELLITE_ENCODER, no_channels=3, dropout=DROPOUT, add_block=EXTRA_LAYER, num_frozen=NUM_FROZEN_LAYERS)
street_encoder = BaseEncoder(arch=STREET_ENCODER, no_channels=3, dropout=DROPOUT, add_block=EXTRA_LAYER, num_frozen=NUM_FROZEN_LAYERS)

# Load Custom Model
model = AQIPrediction(satellite_encoder, street_encoder, dropout=DROPOUT, num_classes=None)  # Initialize your custom model
model.load_state_dict(torch.load("trained_model/resnet18_best_model.pth", map_location=device))
model.to(device)
model.eval()  # Set to evaluation mode

# Image Preprocessing Functions
def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Prediction Function
def predict_aqi(captured_image: Image.Image, satellite_image: Image.Image):
    """Predict AQI using both street and satellite images."""
    captured_tensor = preprocess_image(captured_image)
    satellite_tensor = preprocess_image(satellite_image)

    # Ensure model expects two separate inputs
    with torch.no_grad():
        prediction = model(captured_tensor, satellite_tensor)  # Pass both images properly

    return prediction.item()  # Convert tensor output to a scalar
