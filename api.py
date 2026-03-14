"""
FastAPI backend for AQI prediction with support for concurrent requests.
"""
import asyncio
import base64
import io
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from src.prediction import get_predictor
from src.satellite_utils import fetch_7_bands
from src.data_processing import get_aqi_color

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AQI Prediction API",
    description="API for predicting Air Quality Index using street and satellite images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor on startup."""
    global predictor
    try:
        predictor = get_predictor()
        logger.info("AQI Predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise e

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    latitude: float
    longitude: float
    street_image_base64: str

class PollutantData(BaseModel):
    AQI: float
    PM2_5: float = 0.0
    PM10: float = 0.0
    O3: float = 0.0
    CO: float = 0.0
    SO2: float = 0.0
    NO2: float = 0.0

class ModelPrediction(BaseModel):
    model: str
    predictions: Optional[PollutantData] = None
    error: Optional[str] = None

class PredictionResponse(BaseModel):
    success: bool
    message: str
    latitude: float
    longitude: float
    average_aqi: Optional[float] = None
    aqi_category: Optional[str] = None
    aqi_color: Optional[List[int]] = None
    model_results: List[ModelPrediction] = []
    seven_band_available: bool = False

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    device: str

def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image.convert('RGB')
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")

def get_aqi_category(aqi: float) -> str:
    """Get AQI category based on value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

async def fetch_satellite_images(lat: float, lon: float) -> Optional[np.ndarray]:
    """Fetch 7-band satellite image asynchronously."""
    loop = asyncio.get_event_loop()
    
    try:
        # Run satellite fetching in thread pool
        seven_band_task = loop.run_in_executor(executor, fetch_7_bands, lat, lon)
        seven_band_data = await seven_band_task
        
        if isinstance(seven_band_data, Exception):
            seven_band_data = None
        
        return seven_band_data
        
    except Exception as e:
        logger.error(f"Error fetching satellite images: {e}")
        return None

async def make_prediction(street_image: Image.Image, 
                         satellite_7_bands: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Make prediction asynchronously."""
    loop = asyncio.get_event_loop()
    
    try:
        # Run prediction in thread pool to avoid blocking
        prediction_task = loop.run_in_executor(
            executor, 
            predictor.predict, 
            street_image, 
            satellite_7_bands  # Fixed: removed the None parameter
        )
        
        results = await prediction_task
        return results
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return [{"error": f"Prediction failed: {e}"}]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    return HealthResponse(
        status="healthy",
        models_loaded=len(predictor.models) if predictor.models else 0,
        device=str(predictor.device)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_aqi_endpoint(request: PredictionRequest):
    """Main prediction endpoint."""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        # Decode street image
        street_image = decode_base64_image(request.street_image_base64)
        
        # Fetch 7-band satellite image asynchronously
        satellite_7_bands = await fetch_satellite_images(
            request.latitude, request.longitude
        )
        
        # Make prediction
        prediction_results = await make_prediction(
            street_image, satellite_7_bands
        )
        
        # Process results
        model_predictions = []
        valid_aqi_values = []
        
        for result in prediction_results:
            if "error" in result:
                model_predictions.append(ModelPrediction(
                    model=result.get("model", "unknown"),
                    error=result["error"]
                ))
            else:
                predictions = result.get("predictions", {})
                model_predictions.append(ModelPrediction(
                    model=result["model"],
                    predictions=PollutantData(
                        AQI=predictions.get("AQI", 0.0),
                        PM2_5=predictions.get("PM2.5", 0.0),
                        PM10=predictions.get("PM10", 0.0),
                        O3=predictions.get("O3", 0.0),
                        CO=predictions.get("CO", 0.0),
                        SO2=predictions.get("SO2", 0.0),
                        NO2=predictions.get("NO2", 0.0)
                    )
                ))
                
                # Collect valid AQI values for averaging
                if predictions.get("AQI", 0) > 0:
                    valid_aqi_values.append(predictions["AQI"])
        
        # Calculate average AQI
        average_aqi = None
        aqi_category = None
        aqi_color = None
        
        if valid_aqi_values:
            average_aqi = sum(valid_aqi_values) / len(valid_aqi_values)
            aqi_category = get_aqi_category(average_aqi)
            aqi_color = get_aqi_color(average_aqi)
        
        return PredictionResponse(
            success=True,
            message="Prediction completed successfully",
            latitude=request.latitude,
            longitude=request.longitude,
            average_aqi=average_aqi,
            aqi_category=aqi_category,
            aqi_color=aqi_color,
            model_results=model_predictions,
            seven_band_available=satellite_7_bands is not None
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/predict-upload")
async def predict_aqi_upload(
    latitude: float = Form(...),
    longitude: float = Form(...),
    street_image: UploadFile = File(...)
):
    """Upload endpoint for prediction."""
    global predictor
    
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        # Validate file type
        if not street_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await street_image.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Fetch 7-band satellite image asynchronously
        satellite_7_bands = await fetch_satellite_images(latitude, longitude)
        
        # Make prediction
        prediction_results = await make_prediction(image, satellite_7_bands)
        
        # Process results (same as above)
        model_predictions = []
        valid_aqi_values = []
        
        for result in prediction_results:
            if "error" in result:
                model_predictions.append({
                    "model": result.get("model", "unknown"),
                    "error": result["error"]
                })
            else:
                predictions = result.get("predictions", {})
                model_predictions.append({
                    "model": result["model"],
                    "predictions": {
                        "AQI": predictions.get("AQI", 0.0),
                        "PM2_5": predictions.get("PM2.5", 0.0),
                        "PM10": predictions.get("PM10", 0.0),
                        "O3": predictions.get("O3", 0.0),
                        "CO": predictions.get("CO", 0.0),
                        "SO2": predictions.get("SO2", 0.0),
                        "NO2": predictions.get("NO2", 0.0)
                    }
                })
                
                if predictions.get("AQI", 0) > 0:
                    valid_aqi_values.append(predictions["AQI"])
        
        # Calculate average AQI
        average_aqi = None
        aqi_category = None
        aqi_color = None
        
        if valid_aqi_values:
            average_aqi = sum(valid_aqi_values) / len(valid_aqi_values)
            aqi_category = get_aqi_category(average_aqi)
            aqi_color = get_aqi_color(average_aqi)
        
        return {
            "success": True,
            "message": "Prediction completed successfully",
            "latitude": latitude,
            "longitude": longitude,
            "average_aqi": average_aqi,
            "aqi_category": aqi_category,
            "aqi_color": aqi_color,
            "model_results": model_predictions,
            "seven_band_available": satellite_7_bands is not None
        }
        
    except Exception as e:
        logger.error(f"Upload prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)