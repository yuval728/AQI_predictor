import streamlit as st
# import numpy as np
import pandas as pd
# import geocoder
from PIL import Image
from streamlit_geolocation import streamlit_geolocation
from utils.fetch_satellite import fetch_satellite
import pydeck as pdk
from utils.predict_aqi import predict_aqi

st.set_page_config(layout="wide")

def get_location():
    location = streamlit_geolocation()
    if location:
        return location["latitude"], location["longitude"]
    return None, None


def main():
    st.title("Air Quality Prediction App")
    # 2 columns
    col1, col2 = st.columns(2)
    
    # sidebar camera
    with st.sidebar:
        toggle_camera = 'visible'
        captured_image = st.camera_input("Preview of Street View", key='camera_input', label_visibility=toggle_camera)
        # fetch location
        st.write("Click the button below to fetch your current location.")
        lat, lon = get_location()
    
    if captured_image and lat and lon:
        image = Image.open(captured_image)   
        captured_width, captured_height = image.size     
        with col1:
            st.image(image, caption="Captured Image", use_container_width=True)
        
        # Get current location
        if lat and lon:
            
            # Fetch satellite image
            satellite_image = fetch_satellite(lat, lon)
            satellite_image = Image.open(satellite_image)
            sat_width, sat_height = satellite_image.size
            # st.write(f"Satellite Image Size: {sat_width} x {sat_height}")
            # st.write(f"Captured Image Size: {captured_width} x {captured_height}")

            # Calculate new width while maintaining aspect ratio
            new_width = int((captured_height / sat_height) * sat_width)
            # st.write(f"New Width: {new_width}")
            satellite_image_resized = satellite_image.resize((new_width, captured_height))
            with col2:
                st.image(satellite_image_resized, caption="Satellite Image", use_container_width=True)
            
           
            # print(f"Captured Image Type: {type(image)}")
            # print(f"Satellite Image Type: {type(satellite_image)}")
            aqi_result = predict_aqi(image, satellite_image)
             
            st.write(f"Location: Latitude {lat}, Longitude {lon}, AQI: {aqi_result}")

            df = pd.DataFrame(
                data=[[lat, lon, aqi_result]],
                columns=["lat", "lon", 'aqi'],
            )
            
            # color selection
            if aqi_result < 50:
                color = [0, 255, 0]  # Green
            elif aqi_result < 100:
                color = [255, 255, 0]  # Yellow
            elif aqi_result < 150:
                color = [255, 165, 0]  # Orange
            else:
                color = [255, 0, 0]  # Red

            # Create DataFrame
            df = pd.DataFrame([[lat, lon, aqi_result]], columns=["lat", "lon", "aqi"])

            # Define PyDeck Layer
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["lon", "lat"],
                get_color=color,  # Color based on AQI
                get_radius=200,  # Adjust size
                pickable=True,  # Enable tooltips
            )

            # Define the PyDeck Map with a Tooltip
            view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=0)

            # Create the PyDeck chart with tooltips
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={"text": "AQI: {aqi}"}  # Tooltip displaying AQI value
                )
            )
        else:
            st.error("Could not retrieve location. Ensure location access is enabled.")
    else:
        st.warning("Please capture an image using the camera to proceed.")
if __name__ == "__main__":
    main()
