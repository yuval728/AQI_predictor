import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.resnet_aqi import AQIPrediction
from utils.resnet_base import BaseEncoder
import torch.nn.functional as F
import numpy as np

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA Available:", torch.cuda.is_available())
print("Using device:", device)

# dictionary of model: [model_path, satellite_encoder, street_encoder, dropout, extra_layer, frozen, satellite channels]
model_dict={
    'initial (resnet18)':[ 'trained_model/resnet18_best_model.pth', 'resnet18', 'resnet18', 0.5, True, 0, 3],
    'new (resnet18)':[ 'trained_model/both_resnet18_best_model.pth', 'resnet18', 'resnet18', 0.5, True, 0, 3],
    'resnet18 and 50':[ 'trained_model/sv_resnet50_st_resnet18_best_model.pth', 'resnet18', 'resnet50', 0.5, False, 0, 3],
    # only satellite
    'resnet18 street only':[ 'trained_model/resnet18_street_30_epochs.pth', None, 'resnet18', 0.5, True, 0, None],
    # 7 bands
    'resnet18 7 bands':[ 'trained_model/sv_resnet18_st_resnet18_best_model_7_bands.pth', 'resnet18','resnet18', 0.5, True, 0, 7]
}

models = []

# Load Array of models
for i in model_dict:
    street_encoder = BaseEncoder(arch=model_dict[i][2], no_channels=3, dropout=model_dict[i][3], add_block=model_dict[i][4], num_frozen=model_dict[i][5])
    
    if model_dict[i][1]:
        satellite_encoder = BaseEncoder(arch=model_dict[i][1], no_channels=model_dict[i][6], dropout=model_dict[i][3], add_block=model_dict[i][4], num_frozen=model_dict[i][5])
        # Load Custom Model
        model = AQIPrediction(satellite_encoder, street_encoder, dropout=model_dict[i][3], num_classes=None)  # Initialize your custom model
    else:
        model = AQIPrediction(None, street_encoder, dropout=model_dict[i][3], num_classes=None)

    model.load_state_dict(torch.load(model_dict[i][0], map_location=device))
    model.to(device)
    model.eval()
    models.append(model)

# Single model

# SATELLITE_ENCODER = 'resnet18'
# STREET_ENCODER = 'resnet18'
# DROPOUT = 0.5
# EXTRA_LAYER = True
# NUM_FROZEN_LAYERS = 0

# satellite_encoder = BaseEncoder(arch=SATELLITE_ENCODER, no_channels=3, dropout=DROPOUT, add_block=EXTRA_LAYER, num_frozen=NUM_FROZEN_LAYERS)
# street_encoder = BaseEncoder(arch=STREET_ENCODER, no_channels=3, dropout=DROPOUT, add_block=EXTRA_LAYER, num_frozen=NUM_FROZEN_LAYERS)

# # Load Custom Model
# model = AQIPrediction(satellite_encoder, street_encoder, dropout=DROPOUT, num_classes=None)  # Initialize your custom model
# model.load_state_dict(torch.load("trained_model/resnet18_best_model.pth", map_location=device))
# model.to(device)
# model.eval()  # Set to evaluation mode

image_net_means = [0.485, 0.456, 0.406] 
image_net_stds = [0.229, 0.224, 0.225]

satellite_net_means = [0.18168019890450375, 0.18805927958530722, 0.20592676343591497, 0.20806291225568016, 0.3423790143310607, 0.23654847637549638, 0.17482840221654344]

satellite_net_stds = [0.19048610465575523, 0.19615030016268702, 0.2125846014779801,  0.21476670175116374, 0.347457205638518, 0.2390436189214837, 0.17736793155031446]

# Image Preprocessing Functions
def preprocess_image(image, bands_7=False) -> torch.Tensor:
    if bands_7:
        # For 7-band satellite images (which are numpy arrays)
        if isinstance(image, np.ndarray):
            # Convert numpy array to tensor
            tensor = torch.from_numpy(image).float()
            
            # Ensure correct shape (C, H, W)
            if tensor.ndim == 3 and tensor.shape[2] <= 7:
                tensor = tensor.permute(2, 0, 1)
            
            # Resize the tensor
            tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            
            # Normalize
            for c in range(tensor.shape[0]):
                tensor[c] = (tensor[c] - satellite_net_means[c]) / satellite_net_stds[c]
            
            return tensor.unsqueeze(0).to(device)  # Add batch dimension
    else:
        # For regular images
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
            
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_net_means, std=image_net_stds)
        ])
        
        return transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Prediction Function
def predict_aqi(captured_image: Image.Image, satellite_image: Image.Image, satellite_7_bands: any) -> tuple:
    """Predict AQI using both street and satellite images."""
    captured_tensor = preprocess_image(captured_image)
    satellite_tensor = preprocess_image(satellite_image)
    if satellite_7_bands.any():
        satellite_tensor_7 = preprocess_image(satellite_7_bands, bands_7=True)

    # Multiple models
    predictions = []
    for model, desc in zip(models, model_dict.values()):
        with torch.no_grad():
            if desc[1]:
                if desc[-1] == 7:
                    prediction = model(captured_tensor, satellite_tensor_7)
                else:
                    prediction = model(captured_tensor, satellite_tensor)
            else:
                prediction = model(captured_tensor)
            # prediction = model(captured_tensor, satellite_tensor) 
            predictions.append(prediction.item())

    # zip the model name and the result
    names= list(model_dict.keys())

    accuracy=["some accuracy" for i in range(len(names))]

    return names, predictions, accuracy
