
import os
import numpy as np
import requests, zipfile, io
from IPython.display import Image
import ee  # Install with "pip install earthengine-api --upgrade"

# Authenticate and initialize Earth Engine
ee.Authenticate()  # Requires Earth Engine account
ee.Initialize(project='ee-yuvalmehta728')

# Define parameters
startDate = '2020-01-01'
endDate = '2020-12-31'
image_res = 30  # Resolution in meters
n_pixels = 224  # Number of pixels
# image_res = 10  # Higher resolution (15 meters per pixel)
# n_pixels = 600  # Smaller area (112x112 pixels

# Load Landsat 8 Collection 2, Level 2 Surface Reflectance dataset
landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterDate(startDate, endDate)

# Cloud masking function
def mask_l8_sr(image):
    """Applies cloud masking to Landsat 8 Collection 2 data."""
    cloud_shadow_bitmask = 1 << 4
    clouds_bitmask = 1 << 3

    # Extract QA_PIXEL band
    qa_pixel = image.select('QA_PIXEL')

    # Create a mask based on the bitmask values for clouds and cloud shadows
    mask = qa_pixel.bitwiseAnd(cloud_shadow_bitmask).eq(0) \
           .And(qa_pixel.bitwiseAnd(clouds_bitmask).eq(0))

    # Apply mask and scale optical bands to reflectance
    return image.updateMask(mask).select(['SR_B2', 'SR_B3', 'SR_B4']).multiply(0.0000275).add(-0.2)

# Apply cloud masking
landsat_masked = landsat.map(mask_l8_sr)

# Function to create bounding box
def get_rectangle(point, image_res, n_pixels):
    """Generates a bounding box of image_res * n_pixels meters around a point."""
    length = image_res * n_pixels
    region = point.buffer(length / 2).bounds().getInfo()['coordinates']
    coords = np.array(region)
    coords = [np.min(coords[:, :, 0]), np.min(coords[:, :, 1]), np.max(coords[:, :, 0]), np.max(coords[:, :, 1])]
    return ee.Geometry.Rectangle(coords)

# Function to visualize images
def visualization(point, name, mask=True, vis_params=None):
    """Visualizes Landsat images with or without cloud masking."""
    if vis_params is None:
        vis_params = {'min': 0, 'max': 0.3, 'gamma': 1.4,
                      'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
                    #   'bands': ['B4', 'B3', 'B2'],
                      'dimensions': f'{n_pixels}x{n_pixels}',
                      'format': 'jpg'}

    # Generate bounding box
    rectangle = get_rectangle(point, image_res, n_pixels)

    # Select the appropriate dataset
    image = (landsat_masked if mask else landsat).mean().clip(rectangle)

    # Get the thumbnail URL and download the image
    try:
        thumb_url = image.getThumbUrl(vis_params)
        response = requests.get(thumb_url)
        response.raise_for_status()
        with open(f"{name}.jpg", 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")

# Define the location as a point
oval = ee.Geometry.Point([77.5946, 12.9716])  # Stanford University

# Visualize images with and without cloud masking
# visualization(point=oval, name='oval_no_mask', mask=False)
# Image(filename='oval_no_mask.jpg')  # Display unmasked image

visualization(point=oval, name='pj2_cloud_masking1', mask=True)
Image(filename='pj2_cloud_masking1.jpg')  # Display cloud-masked image


# # sentinel
# import numpy as np
# import requests
# from IPython.display import Image
# import ee  # Install with "pip install earthengine-api --upgrade"

# # Authenticate and initialize Earth Engine
# ee.Authenticate()  # Requires Earth Engine account
# ee.Initialize(project='')

# # Define parameters
# startDate = '2020-01-01'
# endDate = '2020-12-31'
# image_res = 30  # Resolution in meters
# n_pixels = 224  # Number of pixels

# # Load Sentinel-2 Level 1C dataset
# sentinel = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
#     .filterDate(startDate, endDate) \
#     .filterBounds(ee.Geometry.Point(77.5946, 12.9716))  # Filter by region (Bengaluru, India)

# # Cloud masking function for Sentinel-2
# def mask_s2(image):
#     """Applies cloud masking to Sentinel-2 data using QA60 band."""
#     # Get the QA60 band (cloud mask)
#     qa60 = image.select(['QA60'])
    
#     # Cloud mask: bit 10 (clouds) set to 1
#     cloud_mask = qa60.bitwiseAnd(1 << 10).eq(0)

#     # Apply mask and select relevant bands for true color (Red, Green, Blue)
#     return image.updateMask(cloud_mask).select(['B4', 'B3', 'B2'])

# # Apply cloud masking
# sentinel_masked = sentinel.map(mask_s2)

# # Function to create bounding box
# def get_rectangle(point, image_res, n_pixels):
#     """Generates a bounding box of image_res * n_pixels meters around a point."""
#     length = image_res * n_pixels
#     region = point.buffer(length / 2).bounds().getInfo()['coordinates']
#     coords = np.array(region)
#     coords = [np.min(coords[:, :, 0]), np.min(coords[:, :, 1]), np.max(coords[:, :, 0]), np.max(coords[:, :, 1])]
#     return ee.Geometry.Rectangle(coords)

# # Function to visualize images
# def visualization(point, name, mask=True, vis_params=None):
#     """Visualizes Sentinel-2 images with or without cloud masking."""
#     if vis_params is None:
#         vis_params = {'min': 0, 'max': 3000, 'gamma': 1.4,
#                       'bands': ['B4', 'B3', 'B2'],
#                       'dimensions': f'{n_pixels}x{n_pixels}',
#                       'format': 'jpg'}

#     # Generate bounding box
#     rectangle = get_rectangle(point, image_res, n_pixels)

#     # Select the appropriate dataset
#     image = (sentinel_masked if mask else sentinel).mean().clip(rectangle)

#     # Get the thumbnail URL and download the image
#     try:
#         thumb_url = image.getThumbUrl(vis_params)
#         response = requests.get(thumb_url)
#         response.raise_for_status()
#         with open(f"{name}.jpg", 'wb') as f:
#             f.write(response.content)
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching image: {e}")

# # Define the location as a point (Bengaluru, India)
# bengaluru = ee.Geometry.Point(77.5946, 12.9716)  # Latitude and Longitude of Bengaluru

# # Visualize images with and without cloud masking
# visualization(point=bengaluru, name='bengaluru_no_mask', mask=False)
# Image(filename='bengaluru_no_mask.jpg')  # Display unmasked image

# visualization(point=bengaluru, name='bengaluru_cloud_masking', mask=True)
# Image(filename='bengaluru_cloud_masking.jpg')  # Display cloud-masked image
