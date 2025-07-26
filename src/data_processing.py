"""
Data processing utilities for AQI prediction.
"""
import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime
from typing import Tuple, List
from sklearn.impute import KNNImputer
from src.config import KNOWN_LOCATIONS, IMAGENET_MEANS, IMAGENET_STDS, SATELLITE_MEANS, SATELLITE_STDS


class CustomDataset(Dataset):
    """Custom dataset class for loading street and satellite images with multi-label targets."""
    
    def __init__(self, csv_file, satellite_img_dir, street_img_dir, labels, satellite_transform=None, street_transform=None):
        self.data = csv_file
        self.street_img_dir = street_img_dir
        self.satellite_img_dir = satellite_img_dir
        self.labels = labels
        self.satellite_transform = satellite_transform
        self.street_transform = street_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        street_image_path = os.path.join(self.street_img_dir, self.data['Filename'].iloc[idx])
        satellite_image_path = os.path.join(self.satellite_img_dir, self.data['Normalized_Filename'].iloc[idx] + '.npy')
        
        street_image = self.load_image(street_image_path, self.street_transform)
        satellite_image = self.load_npy(satellite_image_path, self.satellite_transform)
        labels = torch.tensor(self.data[self.labels].iloc[idx], dtype=torch.float32)
        
        return street_image, satellite_image, labels
        
    def load_image(self, file_path, transform=None):
        """Load and transform street view image."""
        img = Image.open(file_path)
        if transform:
            img = transform(img)
        return img
    
    def load_npy(self, file_path, transform=None):
        """Load and transform satellite image from numpy array."""
        img = np.load(file_path)
        img = torch.tensor(img, dtype=torch.float32)

        # If the data is in [0, 255], scale it to [0, 1]
        if img.max() > 1.0:  
            img = img / 255.0  

        # Ensure shape (C, H, W) if it's (H, W, C)
        if img.ndimension() == 3 and img.shape[-1] == 7:  
            img = img.permute(2, 0, 1)  
            
        if transform:
            img = transform(img)
        return img


def get_data_transforms():
    """Get data transformation pipelines for training and testing."""
    
    # Development (training/validation) transforms
    data_transforms = {
        'street_dev': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tuple(IMAGENET_MEANS), tuple(IMAGENET_STDS))
        ]),
        'street_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tuple(IMAGENET_MEANS), tuple(IMAGENET_STDS))
        ]),
        'satellite_dev': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(tuple(SATELLITE_MEANS), tuple(SATELLITE_STDS))
        ]),
        'satellite_test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(tuple(SATELLITE_MEANS), tuple(SATELLITE_STDS))
        ])
    }
    
    # Training transforms with augmentation
    satellite_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),
        transforms.Normalize(tuple(SATELLITE_MEANS), tuple(SATELLITE_STDS)),
    ])

    street_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(), 
        transforms.Normalize(tuple(IMAGENET_MEANS), tuple(IMAGENET_STDS)),
    ])
    
    return data_transforms, satellite_transform, street_transform


def prepare_datasets(train_csv, val_csv, test_csv, dataset_path, labels):
    """Prepare train, validation, and test datasets with proper transforms."""
    
    # Apply KNN imputation to combined dataset for consistency
    combined_df = pd.concat([train_csv, val_csv, test_csv])
    knn_imputer = KNNImputer(n_neighbors=5)
    combined_df[['SO2', 'NO2', 'O3', 'CO']] = knn_imputer.fit_transform(combined_df[['SO2', 'NO2', 'O3', 'CO']])
    
    # Split back into separate datasets
    train_csv_clean = combined_df.iloc[:len(train_csv)]
    val_csv_clean = combined_df.iloc[len(train_csv):len(train_csv) + len(val_csv)]
    test_csv_clean = combined_df.iloc[len(train_csv) + len(val_csv):]
    
    # Get transforms
    data_transforms, satellite_transform, street_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = CustomDataset(
        train_csv_clean, 
        os.path.join(dataset_path, 'satellite_images'), 
        os.path.join(dataset_path, 'All_img'), 
        labels, 
        satellite_transform, 
        street_transform
    )
    
    val_dataset = CustomDataset(
        val_csv_clean, 
        os.path.join(dataset_path, 'satellite_images'), 
        os.path.join(dataset_path, 'All_img'), 
        labels, 
        data_transforms['satellite_dev'], 
        data_transforms['street_dev']
    )
    
    test_dataset = CustomDataset(
        test_csv_clean, 
        os.path.join(dataset_path, 'satellite_images'), 
        os.path.join(dataset_path, 'All_img'), 
        labels, 
        data_transforms['satellite_test'], 
        data_transforms['street_test']
    )
    
    return train_dataset, val_dataset, test_dataset


def normalize_filename(filename: str) -> str:
    """Normalize filename by removing extra extensions."""
    splits = filename.split(".")
    return splits[0] + "." + splits[1][:2]


def normalize_filename_non_unique(filename: str) -> str:
    """Normalize filename for non-unique files by removing suffix."""
    idx = filename.rfind("-")
    return filename[:idx]


def map_locations(data: pd.DataFrame, coordinates: List[Tuple[str, float, float]]) -> Tuple[List[float], List[float]]:
    """Map location names to latitude, longitude coordinates."""
    lat = [0] * len(data["Location"])
    lon = [0] * len(data["Location"])
    
    for i, location in enumerate(data["Location"]):
        for j, coord in enumerate(coordinates):
            if location.lower() in coord[0].lower():
                lat[i] = coord[1]
                lon[i] = coord[2]
                break
    
    return lat, lon


def get_date_time(data_point: pd.Series) -> datetime:
    """Convert data point to datetime object."""
    hour, minute = data_point["Hour"].split(":")
    return datetime(
        year=int(data_point["Year"]),
        month=int(data_point["Month"]),
        day=int(data_point["Day"]),
        hour=int(hour),
        minute=int(minute)
    )


def process_dataset(csv_path: str) -> pd.DataFrame:
    """Process the raw dataset by normalizing filenames and adding coordinates."""
    data = pd.read_csv(csv_path)
    
    # Normalize filenames
    data["Normalized_Filename"] = data["Filename"].apply(normalize_filename)
    
    # Split into unique and non-unique data
    unique_data = data[~data["Normalized_Filename"].str.contains(".jp")]
    non_unique_data = data[data["Normalized_Filename"].str.contains(".jp")]
    
    # Process non-unique data
    if not non_unique_data.empty:
        non_unique_data = non_unique_data.copy()
        non_unique_data["Normalized_Filename"] = non_unique_data["Normalized_Filename"].apply(
            normalize_filename_non_unique
        )
    
    # Combine data back
    processed_data = pd.concat([unique_data, non_unique_data], axis=0).sort_index()
    
    # Add coordinates
    lat, lon = map_locations(processed_data, KNOWN_LOCATIONS)
    processed_data["Latitude"] = lat
    processed_data["Longitude"] = lon
    
    return processed_data


def get_aqi_category(aqi_value: float) -> str:
    """Get AQI category based on AQI value."""
    from src.config import AQI_CATEGORIES
    
    for category, (min_val, max_val) in AQI_CATEGORIES.items():
        if min_val <= aqi_value <= max_val:
            return category
    
    return "Hazardous"  # Default for values > 500


def get_aqi_color(aqi_value: float) -> List[int]:
    """Get color for AQI visualization based on AQI value."""
    from src.config import AQI_COLORS
    
    category = get_aqi_category(aqi_value)
    return AQI_COLORS.get(category, AQI_COLORS["Hazardous"])


def impute_missing_values(df: pd.DataFrame, strategy: str = 'knn') -> pd.DataFrame:
    """Impute missing values in the dataset."""
    from sklearn.impute import KNNImputer
    
    if strategy == 'knn':
        # Select numeric columns for imputation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    elif strategy == 'interpolate':
        df.interpolate(method='linear', inplace=True)
    
    return df
