import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.model import BaseEncoder, AQIPrediction
import torch.nn.functional as F
import numpy as np

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA Available:", torch.cuda.is_available())
print("Using device:", device)

# dictionary of model: [model_path, satellite_encoder, street_encoder, dropout, extra_layer, frozen, satellite channels, attention_type]
model_dict = {
    # 'initial (resnet18)': ['trained_model/resnet18_best_model.pth', 'resnet18', 'resnet18', 0.5, True, 0, 3, None],
    # 'new (resnet18)': ['trained_model/both_resnet18_best_model.pth', 'resnet18', 'resnet18', 0.5, True, 0, 3, 'sigmoid_gated'],
    # 'resnet18 and 50': ['trained_model/sv_resnet50_st_resnet18_best_model.pth', 'resnet18', 'resnet50', 0.5, False, 0, 3, 'softmax_gated'],
    # # only satellite
    # 'resnet18 street only': ['trained_model/resnet18_street_30_epochs.pth', None, 'resnet18', 0.5, True, 0, None, None],
    # # 7 bands
    # 'resnet18 7 bands': ['trained_model/sv_resnet18_st_resnet18_best_model_7_bands.pth', 'resnet18', 'resnet18', 0.5, True, 0, 7, 'cross']

    # multi output
    # efficient b0
    'Effientnet b0': ['trained_model/st-efficientnet_b0_sv-efficientnet_b0_attn-sigmoid_gated_best_model.pth', 'efficientnet_b0', 'efficientnet_b0', 0.5, False, 0, 7, 'sigmoid_gated'],
    # efficient b1
    'Effientnet b1': ['trained_model/st-efficientnet_b1_sv-efficientnet_b1_attn-softmax_gated_best_model.pth', 'efficientnet_b1', 'efficientnet_b1', 0.5, False, 0, 7, 'softmax_gated'],
    # efficient b2
    'Effientnet b2': ['trained_model/st-efficientnet_b2_sv-efficientnet_b2_attn-sigmoid_gated_best_model.pth', 'efficientnet_b2', 'efficientnet_b2', 0.5, False, 0, 7, 'sigmoid_gated'],
    # mobile net large
    'MobileNet v3 large': ['trained_model/st-mobilenet_v3_large_sv-mobilenet_v3_large_attn-sigmoid_gated_best_model.pth', 'mobilenet_v3_large', 'mobilenet_v3_large', 0.5, False, 0, 7, 'sigmoid_gated'],
}

models = []

# Load Array of models
for i in model_dict:
    street_encoder = BaseEncoder(
        arch=model_dict[i][2],
        no_channels=3,
        dropout=model_dict[i][3],
        add_block=model_dict[i][4],
        num_frozen=model_dict[i][5]
    )
    
    if model_dict[i][1]:
        satellite_encoder = BaseEncoder(
            arch=model_dict[i][1],
            no_channels=model_dict[i][6],
            dropout=model_dict[i][3],
            add_block=model_dict[i][4],
            num_frozen=model_dict[i][5]
        )
        # Load Custom Model
        model = AQIPrediction(
            satellite_model=satellite_encoder,
            street_model=street_encoder,
            attention_type=model_dict[i][7],
            dropout=model_dict[i][3],
            num_classes=None
        )
    else:
        model = AQIPrediction(
            satellite_model=None,
            street_model=street_encoder,
            attention_type=model_dict[i][7],
            dropout=model_dict[i][3],
            num_classes=None
        )

    model.load_state_dict(torch.load(model_dict[i][0], map_location=device))
    model.to(device)
    model.eval()
    models.append(model)

# Image Preprocessing Functions
image_net_means = [0.485, 0.456, 0.406]
image_net_stds = [0.229, 0.224, 0.225]

satellite_net_means = [0.18168019890450375, 0.18805927958530722, 0.20592676343591497, 0.20806291225568016, 0.3423790143310607, 0.23654847637549638, 0.17482840221654344]
satellite_net_stds = [0.19048610465575523, 0.19615030016268702, 0.2125846014779801, 0.21476670175116374, 0.347457205638518, 0.2390436189214837, 0.17736793155031446]

def preprocess_image(image, bands_7=False) -> torch.Tensor:
    if bands_7:
        # For 7-band satellite images (which are numpy arrays)
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image).float()
            if tensor.ndim == 3 and tensor.shape[2] <= 7:
                tensor = tensor.permute(2, 0, 1)
            tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            for c in range(tensor.shape[0]):
                tensor[c] = (tensor[c] - satellite_net_means[c]) / satellite_net_stds[c]
            return tensor.unsqueeze(0).to(device)
    else:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_net_means, std=image_net_stds)
        ])
        return transform(image).unsqueeze(0).to(device)

# Prediction Function
def predict_aqi(captured_image: Image.Image, satellite_image: Image.Image, satellite_7_bands: any) -> list:
    """Predict AQI, PM2.5, PM10, and gas concentrations using both street and satellite images."""
    captured_tensor = preprocess_image(captured_image) # 3 channel
    satellite_tensor = preprocess_image(satellite_image) # 3 channel
    if satellite_7_bands.any():
        satellite_tensor_7 = preprocess_image(satellite_7_bands, bands_7=True) # 7 channel

    predictions = []
    for model, desc in zip(models, model_dict.values()):
        with torch.no_grad():
            if desc[1]:  # If satellite encoder exists
                if desc[-2] == 7:  # For 7-band satellite images
                    aqi, pm, gas = model(captured_tensor, satellite_tensor_7)
                else:
                    aqi, pm, gas = model(captured_tensor, satellite_tensor)
            else:  # Only street encoder
                aqi, pm, gas = model(captured_tensor)

            predictions.append({
            "AQI": aqi.item(),
            "PM2.5": pm[0, 0].item(),  # Access first element of the batch
            "PM10": pm[0, 1].item(),   # Access second element of the batch
            "O3": gas[0, 0].item(),
            "CO": gas[0, 1].item(),
            "SO2": gas[0, 2].item(),
            "NO2": gas[0, 3].item()
            })

    names = list(model_dict.keys())
    results = [{"model": name, "predictions": pred} for name, pred in zip(names, predictions)]

    return results
