# AQI Predictor

A machine learning application that predicts Air Quality Index (AQI) and air pollutant concentrations using street view and satellite images.

## 🚀 Features

- **Multi-modal prediction**: Combines street view and satellite imagery for accurate AQI prediction
- **Multiple models**: Support for various neural network architectures (EfficientNet, ResNet, MobileNet)
- **7-band satellite data**: Utilizes all Landsat 8 spectral bands for enhanced predictions
- **Real-time prediction**: Web interface for live AQI prediction using camera and GPS
- **Batch processing**: Tools for processing large datasets
- **Modular architecture**: Clean, maintainable codebase following best practices

## 📁 Project Structure

```text
AQI_predictor/
├── src/                          # Core source code
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── data_processing.py        # Data preprocessing utilities
│   ├── losses.py                 # Custom loss functions
│   ├── models.py                 # Neural network models
│   ├── pipeline.py               # Training pipeline
│   ├── prediction.py             # Prediction utilities
│   ├── satellite_utils.py        # Satellite image fetching
│   ├── trainer.py                # Model training logic
│   └── visualization.py          # Plotting and visualization utilities
├── api.py                        # FastAPI backend server
├── app.py                        # Streamlit web application
├── run_pipeline.py               # Script to run the training pipeline
├── requirements.txt              # Core project dependencies
├── api_requirements.txt          # API-specific dependencies
├── ui_requirements.txt           # UI-specific dependencies
├── pyproject.toml                # Project metadata and dependencies (uv)
├── uv.lock                       # uv lockfile
├── README.md                     # This file
└── checkpoints/                  # Pre-trained model weights
    └── st-resnet18_sv-resnet18_attn-sigmoid_gated_best_model.pth
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yuval728/AQI_predictor.git
   cd AQI_predictor
   ```

2. **Install dependencies**:
   Using `uv` (recommended):
   ```bash
   uv sync
   ```
   Or using pip:
   ```bash
   pip install -r requirements.txt
   pip install -r api_requirements.txt
   pip install -r ui_requirements.txt
   ```

3. **Set up Google Earth Engine**:
   - Sign up for Google Earth Engine at https://earthengine.google.com/
   - Install the Earth Engine Python API: `pip install earthengine-api`
   - Authenticate: `earthengine authenticate`
   - Set your project ID in `src/config.py`

## 🚀 Quick Start

### Web Application & API

The application consists of a FastAPI backend and a Streamlit frontend.

1. **Start the FastAPI server**:
   ```bash
   uvicorn api:app --reload
   # or
   python api.py
   ```

2. **Start the Streamlit web application (in a new terminal)**:
   ```bash
   streamlit run app.py
   ```

This will launch a web interface where you can:
- Upload or capture street view images
- Get GPS coordinates automatically
- View predictions from the model
- See satellite imagery and 7-band analysis
- Visualize results on an interactive map

### Model Training

Run the training pipeline:

```bash
python run_pipeline.py
```

## 🏗️ Architecture

### Models

The application supports multiple neural network architectures:

- **EfficientNet B0, B1, B2**: Efficient and accurate models
- **MobileNet V3 Large**: Lightweight model for mobile deployment
- **ResNet 18, 34, 50**: Classic convolutional architectures
- **Vision Transformers**: Transformer-based image models

### Attention Mechanisms

- **Sigmoid Gated Fusion**: Independent attention for each modality
- **Softmax Gated Fusion**: Competing attention between modalities
- **Cross Attention**: Multi-head attention mechanism

### Data Processing

- **Street Images**: RGB images from street-level cameras
- **Satellite Images**: 
  - RGB composite from Landsat 8
  - 7-band spectral data (Blue, Green, Red, NIR, SWIR1, TIR, SWIR2)
- **Cloud Masking**: Automatic cloud removal from satellite imagery
- **Data Augmentation**: Random transformations for training

## 📊 Predicted Metrics

The model predicts the following air quality metrics:

- **AQI**: Air Quality Index (0-500 scale)
- **PM2.5**: Fine particulate matter (μg/m³)
- **PM10**: Coarse particulate matter (μg/m³)
- **O3**: Ozone (μg/m³)
- **CO**: Carbon monoxide (μg/m³)
- **SO2**: Sulfur dioxide (μg/m³)
- **NO2**: Nitrogen dioxide (μg/m³)

## 🗂️ Dataset

The project uses the [Air Pollution Image Dataset from India and Nepal](https://www.kaggle.com/datasets/adarshrouniyar/air-pollution-image-dataset-from-india-and-nepal) which includes:

- Street view images from various locations
- Corresponding air quality measurements
- Temporal data (date/time stamps)
- Location coordinates

## 🔧 Configuration

Key configuration options in `src/config.py`:

- **Model settings**: Architecture, dropout, attention type
- **Image processing**: Resolution, normalization parameters
- **Earth Engine**: Project ID, date ranges
- **Training**: Batch size, learning rate, epochs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Earth Engine for satellite imagery
- Kaggle for the air pollution dataset
- PyTorch and Streamlit communities
- Contributors to the open-source libraries used

---

*Made with ❤️ for cleaner air and better health*
