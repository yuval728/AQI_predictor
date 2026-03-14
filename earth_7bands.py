import os
import numpy as np
import requests
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import ee

ee.Authenticate()
ee.Initialize(project= os.getenv("projectid") )

# parameters
startDate = '2023-01-01'
endDate = '2023-12-31'
image_res = 30  # Resolution in meters
n_pixels = 224  # Number of pixels

# Landsat 8 Collection 2, Level 2 Surface Reflectance dataset
landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterDate(startDate, endDate)

# cloud masking function
def mask_l8_sr(image):
    """Applies cloud masking to Landsat 8 Collection 2 data."""
    cloud_shadow_bitmask = 1 << 4
    clouds_bitmask = 1 << 3

    # Extract QA_PIXEL band
    qa_pixel = image.select('QA_PIXEL')

    # Create a mask based on the bitmask values for clouds and cloud shadows
    mask = qa_pixel.bitwiseAnd(cloud_shadow_bitmask).eq(0) \
           .And(qa_pixel.bitwiseAnd(clouds_bitmask).eq(0))

    # mask and scale optical bands to reflectance
    return image.updateMask(mask).select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']).multiply(0.0000275).add(-0.2)

# cloud masking
landsat_masked = landsat.map(mask_l8_sr)

# Function to create bounding box
def get_rectangle(point, image_res, n_pixels):
    """Generates a bounding box of image_res * n_pixels meters around a point."""
    length = image_res * n_pixels
    region = point.buffer(length / 2).bounds().getInfo()['coordinates']
    coords = np.array(region)
    coords = [np.min(coords[:, :, 0]), np.min(coords[:, :, 1]), np.max(coords[:, :, 0]), np.max(coords[:, :, 1])]
    return ee.Geometry.Rectangle(coords)

# Function to download and process images

# What This Does:
# Downloads the Landsat image as a .tif file.
# Loads the image into a NumPy array.
# Saves the array as a .npy file using np.save().
# Optionally, displays the RGB visualization.
def download_and_process_image(point, name, mask=True):
    """Downloads and processes Landsat images and saves as .npy file."""
    # bounding box
    rectangle = get_rectangle(point, image_res, n_pixels)

    image = (landsat_masked if mask else landsat).mean().clip(rectangle)

    # download URL for the image
    try:
        download_url = image.getDownloadURL({
            'scale': image_res,
            'region': rectangle,
            'format': 'GEO_TIFF'
        })
        response = requests.get(download_url)
        response.raise_for_status()

        # save temp tif file
        tif_path = f"{name}.tif"
        with open(tif_path, 'wb') as f:
            f.write(response.content)

        # loading the image
        with rasterio.open(tif_path) as src:
            array = src.read()  # Shape will be (bands, height, width)
            print(f"Image dimensions: {array.shape}")  # Should be (7, height, width)

            # .npy file
            npy_path = f"{name}.npy"
            np.save(npy_path, array)
            print(f"Saved image as {npy_path}")

            
            # plt.figure(figsize=(10, 10))
            # show(array[[3, 2, 1], :, :], transform=src.transform)  # Bands 4, 3, 2 for RGB
            # plt.title(f"{name} (RGB)")
            # plt.show()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")

# location
oval = ee.Geometry.Point(77.5946, 12.9716)  # Example location

# download tiff and npy
download_and_process_image(point=oval, name='all_bandzz', mask=True)