import os
import numpy as np
import requests
import rasterio
from datetime import datetime, timedelta
from typing import Optional
import ee
from src.config import EARTH_ENGINE_PROJECT_ID, IMAGE_RESOLUTION, IMAGE_PIXELS, DATE_RANGE_DAYS


def initialize_earth_engine():
    """Initialize Google Earth Engine authentication and project."""
    try:
        ee.Authenticate()
        ee.Initialize(project=EARTH_ENGINE_PROJECT_ID)
        return True
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        return False


def mask_landsat_8(image):
    """Apply cloud masking to Landsat 8 Collection 2 data."""
    cloud_shadow_bitmask = 1 << 4
    clouds_bitmask = 1 << 3
    
    qa_pixel = image.select('QA_PIXEL')
    mask = (qa_pixel.bitwiseAnd(cloud_shadow_bitmask).eq(0)
            .And(qa_pixel.bitwiseAnd(clouds_bitmask).eq(0)))
    
    return (image.updateMask(mask)
            .select(['SR_B2', 'SR_B3', 'SR_B4'])
            .multiply(0.0000275)
            .add(-0.2))


def mask_landsat_8_seven_bands(image):
    """Apply cloud masking to Landsat 8 Collection 2 data for 7 bands."""
    cloud_shadow_bitmask = 1 << 4
    clouds_bitmask = 1 << 3
    
    qa_pixel = image.select('QA_PIXEL')
    mask = (qa_pixel.bitwiseAnd(cloud_shadow_bitmask).eq(0)
            .And(qa_pixel.bitwiseAnd(clouds_bitmask).eq(0)))
    
    return (image.updateMask(mask)
            .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
            .multiply(0.0000275)
            .add(-0.2))


def get_rectangle(point, image_res: int = IMAGE_RESOLUTION, n_pixels: int = IMAGE_PIXELS):
    """Generate a bounding box around a point."""
    length = image_res * n_pixels
    region = point.buffer(length / 2).bounds().getInfo()['coordinates']
    coords = np.array(region)
    coords = [
        np.min(coords[:, :, 0]),
        np.min(coords[:, :, 1]),
        np.max(coords[:, :, 0]),
        np.max(coords[:, :, 1])
    ]
    return ee.Geometry.Rectangle(coords)


class SatelliteImageFetcher:
    """Handle satellite image fetching operations."""
    
    def __init__(self):
        self.initialized = initialize_earth_engine()
    
    def fetch_rgb_image(self, lat: float, lon: float, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> Optional[str]:
        """Fetch RGB satellite image for given coordinates."""
        if not self.initialized:
            print("Earth Engine not initialized")
            return None
        
        # Set default date range
        if not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.today() - timedelta(days=DATE_RANGE_DAYS)).strftime('%Y-%m-%d')
        
        try:
            # Create Landsat collection
            landsat = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                      .filterDate(start_date, end_date)
                      .map(mask_landsat_8))
            
            point = ee.Geometry.Point([lon, lat])
            rectangle = get_rectangle(point)
            image = landsat.mean().clip(rectangle)
            
            # Visualization parameters
            vis_params = {
                'min': 0,
                'max': 0.3,
                'gamma': 1.4,
                'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
                'dimensions': f'{IMAGE_PIXELS}x{IMAGE_PIXELS}',
                'format': 'jpg'
            }
            
            # Get thumbnail URL and download
            thumb_url = image.getThumbUrl(vis_params)
            response = requests.get(thumb_url)
            response.raise_for_status()
            
            img_path = "satellite_image.jpg"
            with open(img_path, 'wb') as f:
                f.write(response.content)
            
            return img_path
            
        except Exception as e:
            print(f"Error fetching RGB image: {e}")
            return None
    
    def fetch_seven_band_image(self, lat: float, lon: float,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Optional[np.ndarray]:
        """Fetch 7-band satellite image for given coordinates."""
        if not self.initialized:
            print("Earth Engine not initialized")
            return None
        
        # Set default date range
        if not end_date:
            end_date = datetime.today().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.today() - timedelta(days=DATE_RANGE_DAYS)).strftime('%Y-%m-%d')
        
        try:
            # Create Landsat collection with 7 bands
            landsat = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                      .filterDate(start_date, end_date)
                      .map(mask_landsat_8_seven_bands))
            
            point = ee.Geometry.Point([lon, lat])
            rectangle = get_rectangle(point)
            image = landsat.mean().clip(rectangle)
            
            # Download as GeoTIFF
            download_url = image.getDownloadURL({
                'scale': IMAGE_RESOLUTION,
                'region': rectangle,
                'format': 'GEO_TIFF'
            })
            
            response = requests.get(download_url)
            response.raise_for_status()
            
            tif_path = "temp_image.tif"
            with open(tif_path, 'wb') as f:
                f.write(response.content)
            
            # Read the GeoTIFF file
            with rasterio.open(tif_path) as src:
                array = src.read()  # Shape: (7, height, width)
                print(f"7-band image dimensions: {array.shape}")
            
            # Clean up temporary file
            if os.path.exists(tif_path):
                os.remove(tif_path)
            
            return array
            
        except Exception as e:
            print(f"Error fetching 7-band image: {e}")
            return None
    
    def batch_fetch_images(self, coordinates_list: list, 
                          output_dir: str = "images") -> dict:
        """Batch fetch images for multiple coordinates."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = {}
        errors = []
        
        for i, (name, lat, lon, date_info) in enumerate(coordinates_list):
            try:
                print(f"Processing {name} ({i+1}/{len(coordinates_list)})")
                
                # Create date range around the specific date
                if date_info:
                    target_date = datetime.strptime(date_info, '%Y-%m-%d')
                    start_date = (target_date - timedelta(days=365)).strftime('%Y-%m-%d')
                    end_date = target_date.strftime('%Y-%m-%d')
                else:
                    start_date = end_date = None
                
                # Fetch RGB image
                rgb_path = self.fetch_rgb_image(lat, lon, start_date, end_date)
                
                if rgb_path:
                    # Move to output directory
                    output_path = os.path.join(output_dir, f"{name}.jpg")
                    if os.path.exists("satellite_image.jpg"):
                        os.rename("satellite_image.jpg", output_path)
                    
                    results[name] = {
                        'rgb_path': output_path,
                        'lat': lat,
                        'lon': lon,
                        'success': True
                    }
                else:
                    results[name] = {
                        'success': False,
                        'error': 'Failed to fetch RGB image'
                    }
                    errors.append(i)
                    
            except Exception as e:
                print(f"Error processing {name}: {e}")
                results[name] = {
                    'success': False,
                    'error': str(e)
                }
                errors.append(i)
        
        print(f"Batch processing complete. {len(errors)} errors occurred.")
        return results


# Convenience functions for backward compatibility
def fetch_satellite(lat: float, lon: float) -> Optional[str]:
    """Fetch RGB satellite image (backward compatibility)."""
    fetcher = SatelliteImageFetcher()
    return fetcher.fetch_rgb_image(lat, lon)


def fetch_7_bands(lat: float, lon: float) -> Optional[np.ndarray]:
    """Fetch 7-band satellite image (backward compatibility)."""
    fetcher = SatelliteImageFetcher()
    return fetcher.fetch_seven_band_image(lat, lon)
