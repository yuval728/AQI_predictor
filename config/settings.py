"""
Configuration settings for AQI Predictor project.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Satellite image settings
SATELLITE_CONFIG = {
    "image_resolution": 30,  # Resolution in meters
    "n_pixels": 224,  # Number of pixels
    "bands": ['SR_B2', 'SR_B3', 'SR_B4'],  # Blue, Green, Red bands
    "time_window_days": 30,  # Days to look back for images
    "cloud_cover_threshold": 20,  # Maximum cloud cover percentage
}

# Earth Engine settings
EARTH_ENGINE_CONFIG = {
    "project_id": "ee-yuvalmehta728",  # Replace with your Google Earth Engine project ID
    "collection": "LANDSAT/LC08/C02/T1_L2",
    "scale_factor": 0.0000275,
    "offset": -0.2,
}

# Model settings
MODEL_CONFIG = {
    "default_architecture": "resnet18",
    "input_size": (224, 224),
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "dropout": 0.5,
}

# API settings
API_CONFIG = {
    "max_requests_per_minute": 60,
    "timeout": 30,
    "retry_attempts": 3,
}

# Visualization settings
VIS_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "viridis",
}

# AQI categories for classification
AQI_CATEGORIES = {
    "Good": (0, 50),
    "Moderate": (51, 100),
    "Unhealthy for Sensitive Groups": (101, 150),
    "Unhealthy": (151, 200),
    "Very Unhealthy": (201, 300),
    "Hazardous": (301, 500),
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "aqi_predictor.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# Notebook-specific settings
NOTEBOOK_CONFIG = {
    "kaggle_dataset_path": '/kaggle/input/aqi-dataset/dataset',
    "local_dataset_path": './data',
    "default_seed": 71,
    "wandb_project": "AQI-Detection",
    "num_workers": 2,
    "pin_memory": True,
}

# Training configuration presets
TRAINING_PRESETS = {
    "quick": {
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 1e-3,
    },
    "development": {
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-3,
    },
    "production": {
        "epochs": 45,
        "batch_size": 164,
        "learning_rate": 2e-2,
    }
}
