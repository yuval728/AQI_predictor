"""
Data processing utilities for AQI prediction.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List
from src.config import KNOWN_LOCATIONS


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
