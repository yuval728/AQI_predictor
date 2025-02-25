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
    st.title("Air Quality Prediction using Satellite Images")
    st.markdown("This app predicts the Air Quality Index (AQI) using a combination of captured and satellite images.")
    # 2 columns
    st.subheader("Input Images")
    col1, col2 = st.columns(2)
    
    # sidebar camera
    with st.sidebar:
        # choose to upload image or capture image
        st.write("Choose an option to capture an image.")
        uploaded_image = None
        # upload image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
            # st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        st.divider()
        toggle_camera = 'visible'
        camera_image = st.camera_input("Preview of Street View", key='camera_input', label_visibility=toggle_camera)

        # fetch location
        st.divider()
        col5, col6 = st.columns(2)
        with col5:
            st.write("Click the button to fetch your current location.")

        with col6:
            lat, lon = get_location()
            # button to clear location
            if st.button("Clear Location"):
                lat, lon = None, None
                
        
        if lat and lon:
            st.success("Location fetched successfully.")
        else:
            st.info("Location not captured.")
    
    if (uploaded_image!=None or camera_image) and lat and lon:
        image = uploaded_image if uploaded_image!=None else Image.open(camera_image)  
        captured_width, captured_height = image.size     
        
        with col1:
            st.image(image, caption="Captured Image", use_container_width=True)
        
        # Get current location
        if lat and lon:
            
            # Fetch satellite image
            satellite_image = fetch_satellite(lat, lon)
            satellite_image = Image.open(satellite_image)
            # resizing the image to match the captured image
            satellite_image_resized = satellite_image.resize((captured_width, captured_height))

            with col2:
                st.image(satellite_image_resized, caption="Satellite Image", use_container_width=True)
            
           
            # print(f"Captured Image Type: {type(image)}")
            # print(f"Satellite Image Type: {type(satellite_image)}")
            names, aqi_results, accuracy = predict_aqi(image, satellite_image) 
            print(names, aqi_results, accuracy)

            avg_aqi = round(sum(aqi_results) / len(aqi_results),2)

            data = {
                "Model": names,
                "AQI": aqi_results,
                "Accuracy": accuracy
            }
            
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Location Details")
                st.table({"Latitude": lat, "Longitude": lon})
                st.subheader("Average result AQI")
                st.table({"AQI": avg_aqi })
                # dummy table to display the results obtained from different models
                st.subheader("Model Results")
                df = pd.DataFrame(
                    data=data,
                    columns=["Model", "Accuracy", "AQI"],
                )

                st.table(df)

            with col4:
                st.subheader("Location on map")
                
                # color selection
                if avg_aqi < 50:
                    color = [0, 255, 0]  # Green
                elif avg_aqi < 100:
                    color = [255, 255, 0]  # Yellow
                elif avg_aqi < 150:
                    color = [255, 165, 0]  # Orange
                else:
                    color = [255, 0, 0]  # Red

                # Create DataFrame
                df = pd.DataFrame([[lat, lon, avg_aqi]], columns=["lat", "lon", "aqi"])

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
        st.warning("Please capture an image using the camera and give access to location to proceed.")
if __name__ == "__main__":
    main()
