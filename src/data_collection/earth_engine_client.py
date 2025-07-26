"""
Earth Engine client for satellite data collection.
Handles authentication, initialization, and data collection from Google Earth Engine.
"""
import numpy as np
import requests
import ee
import logging
from typing import Optional, Dict, Any

from config.settings import EARTH_ENGINE_CONFIG, SATELLITE_CONFIG

logger = logging.getLogger(__name__)


class EarthEngineClient:
    """Client for interacting with Google Earth Engine API."""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize Earth Engine client.
        
        Args:
            project_id: Google Earth Engine project ID
        """
        self.project_id = project_id or EARTH_ENGINE_CONFIG['project_id']
        self.collection = EARTH_ENGINE_CONFIG['collection']
        self.scale_factor = EARTH_ENGINE_CONFIG['scale_factor']
        self.offset = EARTH_ENGINE_CONFIG['offset']
        self._initialize()
    
    def _initialize(self):
        """Initialize Earth Engine with authentication."""
        try:
            # Check if already initialized
            try:
                ee.data.getAssetRoots()
            except Exception:
                # Not initialized, need to authenticate and initialize
                ee.Authenticate()
                ee.Initialize(project=self.project_id)
            logger.info("Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            raise
    
    def mask_landsat8_sr(self, image):
        """
        Apply cloud masking to Landsat 8 Collection 2 data.
        
        Args:
            image: Earth Engine image
            
        Returns:
            Masked image with selected bands
        """
        cloud_shadow_bitmask = 1 << 4
        clouds_bitmask = 1 << 3
        qa_pixel = image.select('QA_PIXEL')
        
        mask = qa_pixel.bitwiseAnd(cloud_shadow_bitmask).eq(0) \
               .And(qa_pixel.bitwiseAnd(clouds_bitmask).eq(0))
        
        return image.updateMask(mask).select(SATELLITE_CONFIG['bands']) \
                   .multiply(self.scale_factor).add(self.offset)
    
    def get_bounding_box(self, point, image_res: int, n_pixels: int):
        """
        Generate a bounding box around a point.
        
        Args:
            point: Earth Engine point geometry
            image_res: Image resolution in meters
            n_pixels: Number of pixels
            
        Returns:
            Earth Engine rectangle geometry
        """
        length = image_res * n_pixels
        region = point.buffer(length / 2).bounds().getInfo()['coordinates']
        coords = np.array(region)
        coords = [
            np.min(coords[:, :, 0]),  # min longitude
            np.min(coords[:, :, 1]),  # min latitude
            np.max(coords[:, :, 0]),  # max longitude
            np.max(coords[:, :, 1])   # max latitude
        ]
        return ee.Geometry.Rectangle(coords)
    
    def get_landsat_collection(self, start_date: str, end_date: str):
        """
        Get Landsat collection for specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Earth Engine image collection
        """
        return ee.ImageCollection(self.collection).filterDate(start_date, end_date)
    
    def download_image(self, image, vis_params: Dict[str, Any]) -> Optional[str]:
        """
        Download image from Earth Engine.
        
        Args:
            image: Earth Engine image
            vis_params: Visualization parameters
            
        Returns:
            Path to downloaded image file or None if failed
        """
        try:
            thumb_url = image.getThumbUrl(vis_params)
            response = requests.get(thumb_url, timeout=30)
            response.raise_for_status()
            
            img_path = "satellite_image.jpg"
            with open(img_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Image downloaded successfully: {img_path}")
            return img_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during image download: {e}")
            return None
