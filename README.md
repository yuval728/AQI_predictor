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

```
AQI_predictor/
├── src/                          # Core source code
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── data_processing.py        # Data preprocessing utilities
│   ├── satellite_utils.py        # Satellite image fetching
│   ├── models.py                 # Neural network models
│   ├── prediction.py             # Prediction utilities
│   └── training.py               # Training utilities
├── app.py                        # Streamlit web application
├── train.py                      # Model training script
├── preprocess_data.py            # Data preprocessing script
├── requirements.txt              # Project dependencies
├── README.md                     # This file
└── trained_model/                # Pre-trained model weights
    ├── st-efficientnet_b0_sv-efficientnet_b0_attn-sigmoid_gated_best_model.pth
    ├── st-efficientnet_b1_sv-efficientnet_b1_attn-softmax_gated_best_model.pth
    ├── st-efficientnet_b2_sv-efficientnet_b2_attn-sigmoid_gated_best_model.pth
    └── st-mobilenet_v3_large_sv-mobilenet_v3_large_attn-sigmoid_gated_best_model.pth
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/AQI_predictor.git
   cd AQI_predictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Earth Engine**:
   - Sign up for Google Earth Engine at https://earthengine.google.com/
   - Install the Earth Engine Python API: `pip install earthengine-api`
   - Authenticate: `earthengine authenticate`
   - Set your project ID in `src/config.py`

## 🚀 Quick Start

### Web Application

Run the Streamlit web application:

```bash
streamlit run app.py
```

This will launch a web interface where you can:
- Upload or capture street view images
- Get GPS coordinates automatically
- View predictions from multiple models
- See satellite imagery and 7-band analysis
- Visualize results on an interactive map

### Data Preprocessing

Process raw dataset for training:

```bash
python preprocess_data.py \
    --input_csv data/raw/dataset.csv \
    --output_csv data/processed/processed_dataset.csv \
    --download_satellite \
    --satellite_output_dir data/satellite_images \
    --impute_missing
```

### Model Training

Train a new model:

```bash
python train.py \
    --config efficientnet_b0 \
    --data_path data/processed/processed_dataset.csv \
    --street_img_dir data/street_images \
    --satellite_img_dir data/satellite_images \
    --batch_size 32 \
    --epochs 50 \
    --output_dir trained_models
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

## 📞 Contact

For questions or support, please open an issue or contact [your-email@example.com].

---

*Made with ❤️ for cleaner air and better health*
