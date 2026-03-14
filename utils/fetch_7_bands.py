import os
import numpy as np
import requests
import rasterio
from datetime import datetime, timedelta
import ee

ee.Authenticate()
ee.Initialize(project=os.getenv("projectid"))

def mask_l8_sr(image):
    cloud_shadow_bitmask = 1 << 4
    clouds_bitmask = 1 << 3
    qa_pixel = image.select('QA_PIXEL')
    mask = qa_pixel.bitwiseAnd(cloud_shadow_bitmask).eq(0) \
           .And(qa_pixel.bitwiseAnd(clouds_bitmask).eq(0))
    return image.updateMask(mask).select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(0.0000275).add(-0.2)

def get_rectangle(point, image_res=30, n_pixels=224):
    length = image_res * n_pixels
    region = point.buffer(length / 2).bounds().getInfo()['coordinates']
    coords = np.array(region)
    coords = [np.min(coords[:, :, 0]), np.min(coords[:, :, 1]), np.max(coords[:, :, 0]), np.max(coords[:, :, 1])]
    return ee.Geometry.Rectangle(coords)

def fetch_7_bands(lat, lon, endDate=None, startDate=None):
    if not endDate:
        endDate = datetime.today().strftime('%Y-%m-%d')
    if not startDate:
        startDate = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

    landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
        .filterDate(startDate, endDate).map(mask_l8_sr)

    point = ee.Geometry.Point(lon, lat)
    rectangle = get_rectangle(point)
    image = landsat.mean().clip(rectangle)

    try:
        download_url = image.getDownloadURL({
            'scale': 30,
            'region': rectangle,
            'format': 'GEO_TIFF'
        })
        response = requests.get(download_url)
        response.raise_for_status()

        tif_path = "temp_image.tif"
        with open(tif_path, 'wb') as f:
            f.write(response.content)

        with rasterio.open(tif_path) as src:
            array = src.read()  # Shape will be (7, height, width)
            print(f"Image dimensions: {array.shape}")

        return array

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return None

# # Example usage
# image_array = fetch_7_bands(lat=12.9716, lon=77.5946)
# if image_array is not None:
#     print("Image fetched successfully.")
