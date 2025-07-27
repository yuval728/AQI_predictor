import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_geolocation import streamlit_geolocation
import pydeck as pdk
import requests
import base64
import io
from typing import Optional, Dict, Any

st.set_page_config(layout="wide")

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your FastAPI server URL

def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def check_api_health() -> bool:
    """Check if the FastAPI server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def call_prediction_api(image: Image.Image, lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Call the FastAPI prediction endpoint."""
    try:
        # Convert image to base64
        image_base64 = encode_image_to_base64(image)
        
        # Prepare request data
        payload = {
            "latitude": lat,
            "longitude": lon,
            "street_image_base64": image_base64
        }
        
        # Make API call
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=60  # Allow up to 60 seconds for prediction
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server might be overloaded.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None

def get_location():
    location = streamlit_geolocation()
    if location:
        return location["latitude"], location["longitude"]
    return None, None

def display_prediction_results(api_response: Dict[str, Any]):
    """Display the prediction results from API response."""
    if not api_response.get("success"):
        st.error(f"Prediction failed: {api_response.get('message', 'Unknown error')}")
        return
    
    # Extract data
    avg_aqi = api_response.get("average_aqi")
    aqi_category = api_response.get("aqi_category")
    model_results = api_response.get("model_results", [])
    lat = api_response.get("latitude")
    lon = api_response.get("longitude")
    
    if avg_aqi is None:
        st.error("No valid predictions were generated.")
        return
    
    # Prepare data for display
    model_data = []
    for result in model_results:
        if result.get("predictions"):
            pred = result["predictions"]
            model_data.append({
                "Model": result["model"],
                "AQI": pred.get("AQI", 0.0),
                "PM2.5": pred.get("PM2_5", 0.0),
                "PM10": pred.get("PM10", 0.0),
                "O3": pred.get("O3", 0.0),
                "CO": pred.get("CO", 0.0),
                "SO2": pred.get("SO2", 0.0),
                "NO2": pred.get("NO2", 0.0)
            })
        else:
            st.warning(f"Model {result['model']} failed: {result.get('error', 'Unknown error')}")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Location Details")
        st.table({"Latitude": lat, "Longitude": lon})
        
        st.subheader("Average AQI")
        aqi_color = "green" if avg_aqi <= 50 else "orange" if avg_aqi <= 100 else "red"
        st.markdown(f"**AQI: {avg_aqi}** ({aqi_category})")
        st.markdown(f"<div style='background-color:{aqi_color};padding:10px;border-radius:5px;color:white;text-align:center'><b>{aqi_category}</b></div>", unsafe_allow_html=True)
        
        if model_data:
            st.subheader("Model Results")
            df = pd.DataFrame(model_data)
            st.dataframe(df, use_container_width=True)
    
    with col4:
        st.subheader("Location on Map")
        # Get color from API response or calculate
        aqi_color_rgb = api_response.get("aqi_color", [255, 0, 0])  # Default to red
        
        df_map = pd.DataFrame([[lat, lon, avg_aqi]], columns=["lat", "lon", "aqi"])
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_color=aqi_color_rgb,
            get_radius=200,
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=0)
        st.pydeck_chart(pdk.Deck(
            layers=[layer], 
            initial_view_state=view_state, 
            tooltip={"text": f"AQI: {avg_aqi} ({aqi_category})"}
        ))
        
        # Display satellite data availability
        st.subheader("Data Sources")
        seven_band_available = api_response.get("seven_band_available", False)
        
        st.write(f"🛰️ 7-Band Satellite: {'✅' if seven_band_available else '❌'}")

def main():
    st.title("Air Quality Prediction using Satellite Images")
    st.markdown("This app predicts the Air Quality Index (AQI) using a combination of captured and satellite images.")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ The prediction API is not available. Please ensure the FastAPI server is running.")
        st.info("To start the API server, run: `python api.py` or `uvicorn api:app --reload`")
        return
    else:
        st.success("✅ API server is running")
    
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
            st.write(f"📍 Lat: {lat:.4f}, Lon: {lon:.4f}")
        else:
            st.info("Location not captured.")
    
    # Main prediction logic
    if (uploaded_image or camera_image) and lat and lon:
        image = uploaded_image if uploaded_image else Image.open(camera_image)
        
        with col1:
            st.image(image, caption="Captured Image", use_container_width=True)
        
        with col2:
            st.info("7-band satellite image will be fetched during prediction...")
        
        # Prediction button
        if st.button("🔍 Predict Air Quality", type="primary"):
            with st.spinner("Making prediction... This may take a few moments."):
                # Call the API
                api_response = call_prediction_api(image, lat, lon)
                
                if api_response:
                    # Display results
                    display_prediction_results(api_response)
                else:
                    st.error("Failed to get prediction. Please try again.")
    
    elif not (uploaded_image or camera_image):
        st.warning("📸 Please capture or upload an image to proceed.")
    elif not (lat and lon):
        st.warning("📍 Please allow location access to proceed.")
    else:
        st.info("Please provide both an image and location to get AQI prediction.")

if __name__ == "__main__":
    main()
