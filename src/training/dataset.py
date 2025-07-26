import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os


class CustomDataset(Dataset):
    """
    Custom dataset for AQI prediction with dual input (street and satellite images).
    
    Loads both street view images and satellite images (as .npy files) along with their labels.
    """
    
    def __init__(self, csv_file, satellite_img_dir, street_img_dir, label, 
                 satellite_transform=None, street_transform=None):
        """
        Initialize the dataset.
        
        Args:
            csv_file: DataFrame containing the data
            satellite_img_dir: Directory containing satellite images (.npy files)
            street_img_dir: Directory containing street view images
            label: List of column names to use as labels
            satellite_transform: Transforms to apply to satellite images
            street_transform: Transforms to apply to street images
        """
        self.data = csv_file
        self.street_img_dir = street_img_dir
        self.satellite_img_dir = satellite_img_dir
        self.label = label
        self.satellite_transform = satellite_transform
        self.street_transform = street_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image paths
        street_image_path = os.path.join(self.street_img_dir, self.data['Filename'].iloc[idx])
        satellite_image_path = os.path.join(self.satellite_img_dir, 
                                          self.data['Normalized_Filename'].iloc[idx] + '.npy')
        
        # Load images
        street_image = self.load_image(street_image_path, self.street_transform)
        satellite_image = self.load_npy(satellite_image_path, self.satellite_transform)
        
        # Load labels
        labels = torch.tensor(self.data[self.label].iloc[idx], dtype=torch.float32)
        
        return street_image, satellite_image, labels
        
    def load_image(self, file_path, transform=None):
        """Load and transform a regular image file."""
        img = Image.open(file_path)
        if transform:
            img = transform(img)
        return img
    
    def load_npy(self, file_path, transform=None):
        """Load and transform a numpy array satellite image."""
        img = np.load(file_path)
        img = torch.tensor(img, dtype=torch.float32)

        # Scale from [0, 255] to [0, 1] if needed
        if img.max() > 1.0:  
            img = img / 255.0  

        # Ensure shape is (C, H, W) if it's (H, W, C)
        if img.ndimension() == 3 and img.shape[-1] == 7:  
            img = img.permute(2, 0, 1)  
            
        if transform:
            img = transform(img)
        return img


def create_data_transforms():
    """
    Create standard data transformations for the AQI prediction model.
    """
    from torchvision import transforms
    
    # ImageNet normalization parameters
    image_net_means = [0.485, 0.456, 0.406] 
    image_net_stds = [0.229, 0.224, 0.225]
    
    # Satellite-specific normalization parameters (calculated from 7-band satellite data)
    satellite_net_means = [
        0.18168019890450375, 0.18805927958530722, 0.20592676343591497, 
        0.20806291225568016, 0.3423790143310607, 0.23654847637549638, 
        0.17482840221654344
    ]
    satellite_net_stds = [
        0.19048610465575523, 0.19615030016268702, 0.2125846014779801,
        0.21476670175116374, 0.347457205638518, 0.2390436189214837, 
        0.17736793155031446
    ]
    
    # Training transforms (with augmentation)
    satellite_transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),
        transforms.Normalize(tuple(satellite_net_means), tuple(satellite_net_stds)),
    ])

    street_transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(), 
        transforms.Normalize(tuple(image_net_means), tuple(image_net_stds)),
    ])
    
    # Validation/test transforms (no augmentation)
    satellite_transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(tuple(satellite_net_means), tuple(satellite_net_stds))
    ])

    street_transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(tuple(image_net_means), tuple(image_net_stds))
    ])
    
    return {
        'satellite_train': satellite_transform_train,
        'street_train': street_transform_train,
        'satellite_eval': satellite_transform_eval,
        'street_eval': street_transform_eval
    }


def preprocess_data(train_csv, val_csv, test_csv):
    """
    Preprocess the data by applying KNN imputation for missing values.
    
    Args:
        train_csv: Training DataFrame
        val_csv: Validation DataFrame  
        test_csv: Test DataFrame
        
    Returns:
        Tuple of processed DataFrames
    """
    from sklearn.impute import KNNImputer
    import pandas as pd
    
    # Combine all data for consistent imputation
    combined_df = pd.concat([train_csv, val_csv, test_csv])

    # Apply KNN imputation to gas columns
    knn_imputer = KNNImputer(n_neighbors=5)
    combined_df[['SO2', 'NO2', 'O3', 'CO']] = knn_imputer.fit_transform(
        combined_df[['SO2', 'NO2', 'O3', 'CO']]
    )

    # Split back into separate datasets and reset indices to avoid index misalignment
    train_processed = combined_df.iloc[:len(train_csv)].reset_index(drop=True)
    val_processed = combined_df.iloc[len(train_csv):len(train_csv) + len(val_csv)].reset_index(drop=True)
    test_processed = combined_df.iloc[len(train_csv) + len(val_csv):].reset_index(drop=True)
    
    return train_processed, val_processed, test_processed
