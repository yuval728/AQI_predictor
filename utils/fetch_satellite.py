# import os
import numpy as np
import requests
import ee 
from datetime import datetime, timedelta

ee.Authenticate()  
ee.Initialize(project='')  # Replace with your project ID

def fetch_satellite(lat, lon):
    """Fetches a satellite image for the given latitude and longitude."""
    endDate = datetime.today().strftime('%Y-%m-%d')
    startDate = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    image_res = 30  # Resolution in meters
    n_pixels = 224  # Number of pixels

    landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterDate(startDate, endDate)

    def mask_l8_sr(image):
        """Applies cloud masking to Landsat 8 Collection 2 data."""
        cloud_shadow_bitmask = 1 << 4
        clouds_bitmask = 1 << 3
        qa_pixel = image.select('QA_PIXEL')
        mask = qa_pixel.bitwiseAnd(cloud_shadow_bitmask).eq(0) \
               .And(qa_pixel.bitwiseAnd(clouds_bitmask).eq(0))
        return image.updateMask(mask).select(['SR_B2', 'SR_B3', 'SR_B4']).multiply(0.0000275).add(-0.2)

    landsat_masked = landsat.map(mask_l8_sr)

    def get_rectangle(point, image_res, n_pixels):
        """Generates a bounding box of image_res * n_pixels meters around a point."""
        length = image_res * n_pixels
        region = point.buffer(length / 2).bounds().getInfo()['coordinates']
        coords = np.array(region)
        coords = [np.min(coords[:, :, 0]), np.min(coords[:, :, 1]), np.max(coords[:, :, 0]), np.max(coords[:, :, 1])]
        return ee.Geometry.Rectangle(coords)

    def visualization(point, name, mask=True, vis_params=None):
        """Visualizes Landsat images with or without cloud masking."""
        if vis_params is None:
            vis_params = {'min': 0, 'max': 0.3, 'gamma': 1.4,
                          'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
                          'dimensions': f'{n_pixels}x{n_pixels}',
                          'format': 'jpg'}

        rectangle = get_rectangle(point, image_res, n_pixels)
        image = (landsat_masked if mask else landsat).mean().clip(rectangle)

        try:
            thumb_url = image.getThumbUrl(vis_params)
            response = requests.get(thumb_url)
            response.raise_for_status()
            img_path = f"satellite_image.jpg"
            with open(img_path, 'wb') as f:
                f.write(response.content)
            return img_path
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image: {e}")
            return None

    point = ee.Geometry.Point([lon, lat])
    return visualization(point, name='satellite_image', mask=True)
