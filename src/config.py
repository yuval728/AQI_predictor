import os
import torch

# Earth Engine Configuration
EARTH_ENGINE_PROJECT_ID = os.getenv('EARTH_ENGINE_PROJECT_ID', 'ee-yuvalmehta728')

# Dataset Configuration
DATASET_PATH = os.getenv('DATASET_PATH', 'data')
TRAIN_CSV = os.getenv('TRAIN_CSV', 'data/train_data.csv')
VAL_CSV = os.getenv('VAL_CSV', 'data/val_data.csv')
TEST_CSV = os.getenv('TEST_CSV', 'data/test_data.csv')

# Image Processing Parameters
IMAGE_RESOLUTION = 30  # Resolution in meters
IMAGE_PIXELS = 224    # Number of pixels for square images
DATE_RANGE_DAYS = 30  # Days to look back for satellite images

# Training Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 71  # for reproducibility
BATCH_SIZE = 164  # number of samples processed in one training batch
EPOCHS = 1  # number of training iterations over the entire dataset
LEARNING_RATE = 2e-2  # to update model weights
IMG_SIZE = 224  # input image size
NUM_FROZEN_LAYERS = 2  # 0 indicates that all layers are trainable
DROPOUT = 0.5
LABELS = ['AQI', 'PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']

# Model architectures
SATELLITE_ENCODER = "mobilenet_v3_small"
STREET_ENCODER = "mobilenet_v3_small"
ATTENTION_TYPE = "sigmoid_gated"  # "softmax_gated", "sigmoid_gated", "cross"
SATELLITE_EXTRA_LAYER = False
STREET_EXTRA_LAYER = False

# Model Configuration
MODEL_CONFIG = {
    # 'Effientnet b0': {
    #     'path': 'trained_model/st-efficientnet_b0_sv-efficientnet_b0_attn-sigmoid_gated_best_model.pth',
    #     'satellite_encoder': 'efficientnet_b0',
    #     'street_encoder': 'efficientnet_b0',
    #     'dropout': 0.5,
    #     'extra_layer': False,
    #     'frozen_layers': 0,
    #     'satellite_channels': 7,
    #     'attention_type': 'sigmoid_gated'
    # },
    # 'Effientnet b1': {
    #     'path': 'trained_model/st-efficientnet_b1_sv-efficientnet_b1_attn-softmax_gated_best_model.pth',
    #     'satellite_encoder': 'efficientnet_b1',
    #     'street_encoder': 'efficientnet_b1',
    #     'dropout': 0.5,
    #     'extra_layer': False,
    #     'frozen_layers': 0,
    #     'satellite_channels': 7,
    #     'attention_type': 'softmax_gated'
    # },
    'Resnet 18': {
        'path': 'trained_model/st-resnet_18_sv-resnet_18_attn-sigmoid_gated_best_model.pth',
        'satellite_encoder': 'resnet_18',
        'street_encoder': 'resnet_18',
        'dropout': 0.5,
        'extra_layer': False,
        'frozen_layers': 0,
        'satellite_channels': 7,
        'attention_type': 'sigmoid_gated'
    },
    # 'MobileNet v3 large': {
    #     'path': 'trained_model/st-mobilenet_v3_large_sv-mobilenet_v3_large_attn-sigmoid_gated_best_model.pth',
    #     'satellite_encoder': 'mobilenet_v3_large',
    #     'street_encoder': 'mobilenet_v3_large',
    #     'dropout': 0.5,
    #     'extra_layer': False,
    #     'frozen_layers': 0,
    #     'satellite_channels': 7,
    #     'attention_type': 'sigmoid_gated'
    # }
    # 'MobileNet v3 Small': {
    #     'path': 'checkpoints/best_model_mobilenet_v3_small_mobilenet_v3_small.pth',
    #     'satellite_encoder': 'mobilenet_v3_small',
    #     'street_encoder': 'mobilenet_v3_small',
    #     'dropout': 0.5,
    #     'extra_layer': False,
    #     'frozen_layers': 0,
    #     'satellite_channels': 7,
    #     'attention_type': 'sigmoid_gated'
    # }
}

# Normalization constants
IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]

SATELLITE_MEANS = [
    0.18168019890450375, 0.18805927958530722, 0.20592676343591497,
    0.20806291225568016, 0.3423790143310607, 0.23654847637549638,
    0.17482840221654344
]

SATELLITE_STDS = [
    0.19048610465575523, 0.19615030016268702, 0.2125846014779801,
    0.21476670175116374, 0.347457205638518, 0.2390436189214837,
    0.17736793155031446
]

# AQI Categories
AQI_CATEGORIES = {
    'Good': (0, 50),
    'Moderate': (51, 100),  
    'Unhealthy for Sensitive Groups': (101, 150),
    'Unhealthy': (151, 200),
    'Very Unhealthy': (201, 300),
    'Hazardous': (301, 500)
}

# AQI Color mapping for visualization
AQI_COLORS = {
    'Good': [0, 255, 0],
    'Moderate': [255, 255, 0],
    'Unhealthy for Sensitive Groups': [255, 165, 0],
    'Unhealthy': [255, 0, 0],
    'Very Unhealthy': [128, 0, 128],
    'Hazardous': [128, 0, 0]
}

# Coordinate mappings for known locations
KNOWN_LOCATIONS = [
    ("ITO, Delhi", 28.6284, 77.2425),
    ("Dimapur, Nagaland", 25.9094, 93.7276),
    ("Spice Garden, Bengaluru", 12.9724, 77.5807),
    ("Knowledge Park, Greater Noida", 28.4674, 77.503),
    ("New Ind Town, Faridabad", 28.4089, 77.3178),
    ("Borivali East, Mumbai", 19.2300, 72.8602),
    ("Oragadam, Tamil Nadu", 12.8321, 79.9943),
    ("Biratnagar, Nepal", 26.4525, 87.2718)
]
