"""
Satellite data fetcher for AQI prediction.
Handles satellite image collection and preprocessing.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from .earth_engine_client import EarthEngineClient
from config.settings import SATELLITE_CONFIG

logger = logging.getLogger(__name__)


class SatelliteFetcher:
    """Fetches satellite images for given coordinates."""
    
    def __init__(self, ee_client: Optional[EarthEngineClient] = None):
        """
        Initialize satellite fetcher.
        
        Args:
            ee_client: Earth Engine client instance
        """
        self.ee_client = ee_client or EarthEngineClient()
        self.image_res = SATELLITE_CONFIG['image_resolution']
        self.n_pixels = SATELLITE_CONFIG['n_pixels']
        self.bands = SATELLITE_CONFIG['bands']
        self.time_window_days = SATELLITE_CONFIG['time_window_days']
    
    def get_date_range(self, days_back: Optional[int] = None) -> tuple[str, str]:
        """
        Get date range for satellite image search.
        
        Args:
            days_back: Number of days to look back from today
            
        Returns:
            Tuple of (start_date, end_date) in YYYY-MM-DD format
        """
        days_back = days_back or self.time_window_days
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days_back)
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    def get_visualization_params(self, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get visualization parameters for satellite images.
        
        Args:
            custom_params: Custom visualization parameters
            
        Returns:
            Visualization parameters dictionary
        """
        default_params = {
            'min': 0,
            'max': 0.3,
            'gamma': 1.4,
            'bands': ['SR_B4', 'SR_B3', 'SR_B2'],  # RGB composite
            'dimensions': f'{self.n_pixels}x{self.n_pixels}',
            'format': 'jpg'
        }
        
        if custom_params:
            default_params.update(custom_params)
        
        return default_params
    
    def fetch_satellite_image(self, 
                            lat: float, 
                            lon: float, 
                            days_back: Optional[int] = None,
                            apply_cloud_mask: bool = True,
                            vis_params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Fetch satellite image for given coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            days_back: Number of days to look back
            apply_cloud_mask: Whether to apply cloud masking
            vis_params: Custom visualization parameters
            
        Returns:
            Path to downloaded image file or None if failed
        """
        try:
            logger.info(f"Fetching satellite image for coordinates: ({lat}, {lon})")
            
            # Get date range
            start_date, end_date = self.get_date_range(days_back)
            logger.debug(f"Date range: {start_date} to {end_date}")
            
            # Get Landsat collection
            landsat_collection = self.ee_client.get_landsat_collection(start_date, end_date)
            
            # Apply cloud masking if requested
            if apply_cloud_mask:
                landsat_collection = landsat_collection.map(self.ee_client.mask_landsat8_sr)
            
            # Create point geometry
            import ee
            point = ee.Geometry.Point([lon, lat])
            
            # Get bounding box
            rectangle = self.ee_client.get_bounding_box(point, self.image_res, self.n_pixels)
            
            # Get mean image over the time period and clip to rectangle
            image = landsat_collection.mean().clip(rectangle)
            
            # Get visualization parameters
            visualization_params = self.get_visualization_params(vis_params)
            
            # Download image
            image_path = self.ee_client.download_image(image, visualization_params)
            
            if image_path:
                logger.info(f"Successfully fetched satellite image: {image_path}")
            else:
                logger.warning("Failed to fetch satellite image")
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error fetching satellite image: {e}")
            return None
    
    def fetch_multiple_dates(self, 
                           lat: float, 
                           lon: float, 
                           date_ranges: list[tuple[str, str]],
                           apply_cloud_mask: bool = True) -> list[str]:
        """
        Fetch satellite images for multiple date ranges.
        
        Args:
            lat: Latitude
            lon: Longitude
            date_ranges: List of (start_date, end_date) tuples
            apply_cloud_mask: Whether to apply cloud masking
            
        Returns:
            List of paths to downloaded image files
        """
        image_paths = []
        
        for i, (start_date, end_date) in enumerate(date_ranges):
            try:
                logger.info(f"Fetching image {i+1}/{len(date_ranges)} for date range: {start_date} to {end_date}")
                
                # Get Landsat collection for specific date range
                landsat_collection = self.ee_client.get_landsat_collection(start_date, end_date)
                
                if apply_cloud_mask:
                    landsat_collection = landsat_collection.map(self.ee_client.mask_landsat8_sr)
                
                # Create point geometry
                import ee
                point = ee.Geometry.Point([lon, lat])
                rectangle = self.ee_client.get_bounding_box(point, self.image_res, self.n_pixels)
                
                # Get mean image and clip
                image = landsat_collection.mean().clip(rectangle)
                
                # Download with unique filename
                vis_params = self.get_visualization_params()
                vis_params['format'] = 'jpg'
                
                image_path = self.ee_client.download_image(image, vis_params)
                
                if image_path:
                    # Rename to include date info
                    import os
                    base, ext = os.path.splitext(image_path)
                    dated_path = f"{base}_{start_date}_to_{end_date}{ext}"
                    os.rename(image_path, dated_path)
                    image_paths.append(dated_path)
                    logger.info(f"Successfully fetched image: {dated_path}")
                else:
                    logger.warning(f"Failed to fetch image for date range: {start_date} to {end_date}")
                
            except Exception as e:
                logger.error(f"Error fetching image for date range {start_date} to {end_date}: {e}")
        
        return image_paths


def fetch_satellite(lat: float, lon: float) -> Optional[str]:
    """
    Convenience function to fetch satellite image for given coordinates.
    Maintains backward compatibility with existing code.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Path to downloaded image file or None if failed
    """
    fetcher = SatelliteFetcher()
    return fetcher.fetch_satellite_image(lat, lon)
