import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_geolocation import streamlit_geolocation
from utils.fetch_satellite import fetch_satellite
from utils.fetch_7_bands import fetch_7_bands
import pydeck as pdk
from utils.predict_aqi import predict_aqi
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def get_location():
    location = streamlit_geolocation()
    if location:
        return location["latitude"], location["longitude"]
    return None, None

def main():
    st.title("Air Quality Prediction using Satellite Images")
    st.markdown("This app predicts the Air Quality Index (AQI) using a combination of captured and satellite images.")
    st.subheader("Input Images")
    col1, col2 = st.columns(2)
    
    # Sidebar for image input and location
    with st.sidebar:
        st.write("Choose an option to capture an image.")
        uploaded_image = None
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)

        st.divider()
        toggle_camera = 'visible'
        camera_image = st.camera_input("Preview of Street View", key='camera_input', label_visibility=toggle_camera)

        st.divider()
        col5, col6 = st.columns(2)
        with col5:
            st.write("Click the button to fetch your current location.")
        with col6:
            lat, lon = get_location()
            if st.button("Clear Location"):
                lat, lon = None, None

        if lat and lon:
            st.success("Location fetched successfully.")
        else:
            st.info("Location not captured.")
    
    if (uploaded_image or camera_image) and lat and lon:
        image = uploaded_image if uploaded_image else Image.open(camera_image)
        captured_width, captured_height = image.size

        with col1:
            st.image(image, caption="Captured Image", use_container_width=True)

        if lat and lon:
            # Fetch satellite images
            satellite_image = fetch_satellite(lat, lon)
            satellite_image_7_bands = fetch_7_bands(lat, lon)
            satellite_image = Image.open(satellite_image)
            satellite_image_resized = satellite_image.resize((captured_width, captured_height))

            with col2:
                st.image(satellite_image_resized, caption="Satellite Image", use_container_width=True)

            band_names = [
                "Blue", "Green", "Red", "Near Infrared (NIR)",
                "Shortwave Infrared 1 (SWIR 1)", "Thermal Infrared (TIR)", "Shortwave Infrared 2 (SWIR 2)"
            ]

            if satellite_image_7_bands is not None:
                st.subheader("Satellite Image Bands")
                plt.figure(figsize=(15, 3))
                for i, band in enumerate(satellite_image_7_bands):
                    plt.subplot(1, 7, i + 1)
                    plt.imshow(band, cmap='gray')
                    plt.axis('off')
                    plt.title(band_names[i], fontsize=8)
                st.pyplot(plt)
            else:
                st.warning("Could not fetch 7-band satellite image.")

            # Predict AQI
            results = predict_aqi(image, satellite_image, satellite_image_7_bands)

            # Process predictions
            model_names = [result["model"] for result in results]
            aqi_values = [result["predictions"]["AQI"] for result in results]
            avg_aqi = round(sum(aqi_values) / len(aqi_values), 2)

            data = {
                "Model": model_names,
                "AQI": aqi_values,
                "PM2.5": [result["predictions"]["PM2.5"] for result in results],
                "PM10": [result["predictions"]["PM10"] for result in results],
                "O3": [result["predictions"]["O3"] for result in results],
                "CO": [result["predictions"]["CO"] for result in results],
                "SO2": [result["predictions"]["SO2"] for result in results],
                "NO2": [result["predictions"]["NO2"] for result in results]
            }

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Location Details")
                st.table({"Latitude": lat, "Longitude": lon})
                st.subheader("Average AQI")
                st.table({"AQI": avg_aqi})
                st.subheader("Model Results")
                df = pd.DataFrame(data)
                st.table(df)

            with col4:
                st.subheader("Location on Map")
                color = [0, 255, 0] if avg_aqi < 50 else [255, 255, 0] if avg_aqi < 100 else [255, 165, 0] if avg_aqi < 150 else [255, 0, 0]
                df_map = pd.DataFrame([[lat, lon, avg_aqi]], columns=["lat", "lon", "aqi"])
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position=["lon", "lat"],
                    get_color=color,
                    get_radius=200,
                    pickable=True,
                )
                view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=0)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "AQI: {aqi}"}))
        else:
            st.error("Could not retrieve location. Ensure location access is enabled.")
    else:
        st.warning("Please capture an image using the camera and give access to location to proceed.")

if __name__ == "__main__":
    main()
