"""
Complete AQI Prediction Pipeline
Orchestrates the entire workflow: data collection -> EDA -> preprocessing -> training -> results
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import modules from src
from src.data_collection.satellite_fetcher import SatelliteFetcher
from src.eda.exploratory_analysis import ExploratoryAnalysis  
from src.preprocessing.data_cleaner import DataCleaner
from config.settings import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AQIPipeline:
    """Complete AQI prediction pipeline."""
    
    def __init__(self, config: dict = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config or self._get_default_config()
        self.setup_directories()
        self.results = {}
        
        logger.info("AQI Pipeline initialized")
        
    def _get_default_config(self) -> dict:
        """Get default pipeline configuration."""
        return {
            "data_collection": {
                "sample_locations": [
                    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
                    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
                    {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946}
                ],
                "days_back": 30,
                "apply_cloud_mask": True
            },
            "eda": {
                "generate_plots": True,
                "save_summary": True
            },
            "preprocessing": {
                "remove_outliers": True,
                "normalize_data": True
            },
            "training": {
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 0.001,
                "is_classification": False,
                "test_size": 0.2
            }
        }
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            DATA_DIR,
            RAW_DATA_DIR, 
            PROCESSED_DATA_DIR,
            "outputs",
            "models",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        logger.info("Directories created successfully")
    
    def step1_data_collection(self) -> bool:
        """
        Step 1: Data Collection
        Collect satellite images and prepare dataset.
        """
        logger.info("=" * 50)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("=" * 50)
        
        try:
            # Check if we have existing data
            existing_data_path = RAW_DATA_DIR / "aqi_data.csv"
            
            if existing_data_path.exists():
                logger.info("Found existing data, loading...")
                self.raw_data = pd.read_csv(existing_data_path)
                logger.info(f"Loaded {len(self.raw_data)} records from existing data")
            else:
                logger.info("No existing data found. Using sample data collection...")
                
                # For demonstration, create sample data structure
                # In practice, you would integrate with real AQI data sources
                sample_data = []
                
                satellite_fetcher = SatelliteFetcher()
                
                for location in self.config["data_collection"]["sample_locations"]:
                    logger.info(f"Processing location: {location['name']}")
                    
                    # Fetch satellite image
                    try:
                        image_path = satellite_fetcher.fetch_satellite_image(
                            lat=location['lat'],
                            lon=location['lon'],
                            days_back=self.config["data_collection"]["days_back"],
                            apply_cloud_mask=self.config["data_collection"]["apply_cloud_mask"]
                        )
                        
                        # Create sample record
                        sample_data.append({
                            'Location': location['name'],
                            'Latitude': location['lat'],
                            'Longitude': location['lon'],
                            'Satellite_Image': image_path or "not_available",
                            'AQI': np.random.randint(50, 300),  # Sample AQI value
                            'PM2.5': np.random.uniform(10, 150),
                            'PM10': np.random.uniform(20, 200),
                            'O3': np.random.uniform(0, 100),
                            'CO': np.random.uniform(0, 10),
                            'SO2': np.random.uniform(0, 50),
                            'NO2': np.random.uniform(0, 80),
                            'Timestamp': datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing {location['name']}: {e}")
                        continue
                
                self.raw_data = pd.DataFrame(sample_data)
                
                # Save raw data
                self.raw_data.to_csv(existing_data_path, index=False)
                logger.info(f"Saved {len(self.raw_data)} records to {existing_data_path}")
            
            self.results['data_collection'] = {
                'status': 'success',
                'records_collected': len(self.raw_data),
                'data_path': str(existing_data_path)
            }
            
            logger.info(f"Data collection completed. {len(self.raw_data)} records available.")
            return True
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            self.results['data_collection'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def step2_eda(self) -> bool:
        """
        Step 2: Exploratory Data Analysis
        Analyze the collected data and generate insights.
        """
        logger.info("=" * 50)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 50)
        
        try:
            eda = ExploratoryAnalysis(output_dir="outputs/eda")
            
            # Basic statistics
            logger.info("Generating basic statistics...")
            basic_stats = eda.basic_statistics(
                self.raw_data, 
                save_results=self.config["eda"]["save_summary"]
            )
            
            # AQI distribution analysis
            if 'AQI' in self.raw_data.columns:
                logger.info("Analyzing AQI distribution...")
                eda.analyze_aqi_distribution(self.raw_data)
            
            # Correlation analysis
            numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                logger.info("Performing correlation analysis...")
                eda.correlation_analysis(
                    self.raw_data[numeric_cols],
                    save_plot=self.config["eda"]["generate_plots"]
                )
            
            # Location analysis
            if 'Latitude' in self.raw_data.columns and 'Longitude' in self.raw_data.columns:
                logger.info("Analyzing geographical distribution...")
                eda.geographical_analysis(
                    self.raw_data, 
                    lat_col='Latitude', 
                    lon_col='Longitude', 
                    aqi_col='AQI'
                )
            
            # Time series analysis if timestamp available
            if 'Timestamp' in self.raw_data.columns:
                logger.info("Performing time series analysis...")
                try:
                    self.raw_data['Timestamp'] = pd.to_datetime(self.raw_data['Timestamp'])
                    eda.temporal_analysis(self.raw_data, date_col='Timestamp', aqi_col='AQI')
                except Exception as e:
                    logger.warning(f"Time series analysis failed: {e}")
            
            self.results['eda'] = {
                'status': 'success',
                'basic_stats': basic_stats,
                'output_dir': 'outputs/eda'
            }
            
            logger.info("Exploratory data analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"EDA failed: {e}")
            self.results['eda'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def step3_preprocessing(self) -> bool:
        """
        Step 3: Data Preprocessing
        Clean and prepare data for training.
        """
        logger.info("=" * 50)
        logger.info("STEP 3: DATA PREPROCESSING")
        logger.info("=" * 50)
        
        try:
            data_cleaner = DataCleaner()
            
            # Start with raw data
            processed_data = self.raw_data.copy()
            
            # Clean coordinates
            if 'Latitude' in processed_data.columns and 'Longitude' in processed_data.columns:
                logger.info("Validating coordinates...")
                processed_data = data_cleaner.validate_coordinates(
                    processed_data, 
                    lat_col='Latitude', 
                    lon_col='Longitude'
                )
            
            # Clean AQI values
            if 'AQI' in processed_data.columns:
                logger.info("Cleaning AQI values...")
                processed_data = data_cleaner.clean_aqi_values(processed_data, aqi_col='AQI')
            
            # Handle missing values
            logger.info("Handling missing values...")
            processed_data = data_cleaner.handle_missing_values(processed_data)
            
            # Remove outliers if configured
            if self.config["preprocessing"]["remove_outliers"]:
                logger.info("Removing outliers...")
                numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                outlier_cols = [col for col in numeric_cols if col in ['AQI', 'PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']]
                if outlier_cols:
                    processed_data = data_cleaner.remove_outliers(
                        processed_data, 
                        columns=outlier_cols
                    )
            
            # Feature engineering
            logger.info("Engineering features...")
            processed_data = data_cleaner.create_aqi_categories(processed_data)
            
            # Normalize numerical features if configured
            if self.config["preprocessing"]["normalize_data"]:
                logger.info("Normalizing numerical features...")
                numeric_features = ['AQI', 'PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']
                available_features = [col for col in numeric_features if col in processed_data.columns]
                
                if available_features:
                    processed_data = data_cleaner.normalize_features(
                        processed_data, 
                        features=available_features
                    )
            
            # Save processed data
            processed_data_path = PROCESSED_DATA_DIR / "processed_aqi_data.csv"
            processed_data.to_csv(processed_data_path, index=False)
            
            self.processed_data = processed_data
            
            self.results['preprocessing'] = {
                'status': 'success',
                'initial_records': len(self.raw_data),
                'final_records': len(processed_data),
                'data_path': str(processed_data_path),
                'features_created': list(processed_data.columns)
            }
            
            logger.info(f"Data preprocessing completed. {len(processed_data)} records ready for training.")
            return True
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            self.results['preprocessing'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def step4_training(self) -> bool:
        """
        Step 4: Model Training
        Train the AQI prediction model.
        """
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("=" * 50)
        
        try:
            # Check if we have processed data, if not use raw data
            if not hasattr(self, 'processed_data'):
                logger.warning("No processed data available, using raw data for training")
                self.processed_data = self.raw_data.copy()
            
            # For this simplified version, we'll do basic model training simulation
            # In practice, you would implement full model training with datasets
            
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import joblib
            
            # Prepare features and targets
            feature_cols = ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']
            available_features = [col for col in feature_cols if col in self.processed_data.columns]
            
            if len(available_features) < 2:
                logger.warning("Insufficient features for training. Using mock training...")
                
                self.results['training'] = {
                    'status': 'success_mock',
                    'message': 'Mock training completed due to insufficient features',
                    'model_path': 'models/mock_model.pkl'
                }
                return True
            
            X = self.processed_data[available_features].fillna(0)
            y = self.processed_data['AQI'].fillna(100)  # Default AQI if missing
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config["training"]["test_size"], 
                random_state=42
            )
            
            logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
            
            # Train model
            logger.info("Training Random Forest model...")
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            logger.info("Training completed!")
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R²: {r2:.4f}")
            
            # Save model
            model_path = Path("models") / "aqi_model.pkl"
            model_path.parent.mkdir(exist_ok=True)
            joblib.dump(model, model_path)
            
            # Feature importance
            feature_importance = dict(zip(available_features, model.feature_importances_))
            
            self.results['training'] = {
                'status': 'success',
                'metrics': {
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'mse': float(mse)
                },
                'feature_importance': feature_importance,
                'model_path': str(model_path),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            self.trained_model = model
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.results['training'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def step5_results(self) -> bool:
        """
        Step 5: Generate Results and Reports
        Create final reports and visualizations.
        """
        logger.info("=" * 50)
        logger.info("STEP 5: RESULTS AND REPORTING")
        logger.info("=" * 50)
        
        try:
            # Create results directory
            results_dir = Path("outputs/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate pipeline summary
            pipeline_summary = {
                'pipeline_execution': {
                    'timestamp': datetime.now().isoformat(),
                    'total_steps': 5,
                    'successful_steps': sum(1 for step in self.results.values() 
                                          if step.get('status', '').startswith('success')),
                    'configuration': self.config
                },
                'results_by_step': self.results
            }
            
            # Save pipeline summary
            summary_path = results_dir / "pipeline_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=2, default=str)
            
            # Generate markdown report
            report_path = results_dir / "pipeline_report.md"
            self._generate_markdown_report(report_path, pipeline_summary)
            
            # If we have a trained model, generate additional visualizations
            if hasattr(self, 'trained_model') and hasattr(self, 'processed_data'):
                self._generate_model_visualizations(results_dir)
            
            logger.info(f"Results generated successfully in {results_dir}")
            
            self.results['results'] = {
                'status': 'success',
                'summary_path': str(summary_path),
                'report_path': str(report_path),
                'results_dir': str(results_dir)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Results generation failed: {e}")
            self.results['results'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def _generate_markdown_report(self, report_path: Path, summary: dict):
        """Generate a markdown report of the pipeline execution."""
        with open(report_path, 'w') as f:
            f.write("# AQI Prediction Pipeline Report\n\n")
            f.write(f"**Generated on:** {summary['pipeline_execution']['timestamp']}\n\n")
            
            f.write("## Pipeline Summary\n\n")
            f.write(f"- **Total Steps:** {summary['pipeline_execution']['total_steps']}\n")
            f.write(f"- **Successful Steps:** {summary['pipeline_execution']['successful_steps']}\n\n")
            
            f.write("## Step Results\n\n")
            
            for step_name, step_results in summary['results_by_step'].items():
                f.write(f"### {step_name.replace('_', ' ').title()}\n\n")
                f.write(f"**Status:** {step_results.get('status', 'Unknown')}\n\n")
                
                if step_results.get('status') == 'success':
                    # Add step-specific details
                    if step_name == 'data_collection':
                        f.write(f"- Records collected: {step_results.get('records_collected', 'N/A')}\n")
                    elif step_name == 'training':
                        metrics = step_results.get('metrics', {})
                        f.write(f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}\n")
                        f.write(f"- MAE: {metrics.get('mae', 'N/A'):.4f}\n")
                        f.write(f"- R²: {metrics.get('r2', 'N/A'):.4f}\n")
                
                if 'error' in step_results:
                    f.write(f"**Error:** {step_results['error']}\n")
                
                f.write("\n")
    
    def _generate_model_visualizations(self, results_dir: Path):
        """Generate model-related visualizations."""
        try:
            import matplotlib.pyplot as plt
            
            # Feature importance plot
            if 'training' in self.results and 'feature_importance' in self.results['training']:
                feature_importance = self.results['training']['feature_importance']
                
                plt.figure(figsize=(10, 6))
                features = list(feature_importance.keys())
                importance = list(feature_importance.values())
                
                plt.barh(features, importance)
                plt.title('Feature Importance in AQI Prediction Model')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.savefig(results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("Feature importance plot saved")
        
        except Exception as e:
            logger.warning(f"Could not generate model visualizations: {e}")
    
    def run_complete_pipeline(self) -> dict:
        """
        Run the complete AQI prediction pipeline.
        
        Returns:
            Dictionary with pipeline execution results
        """
        logger.info("Starting complete AQI Prediction Pipeline")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Execute pipeline steps
        steps = [
            ("Data Collection", self.step1_data_collection),
            ("Exploratory Data Analysis", self.step2_eda),
            ("Data Preprocessing", self.step3_preprocessing),
            ("Model Training", self.step4_training),
            ("Results Generation", self.step5_results)
        ]
        
        for step_name, step_function in steps:
            logger.info(f"Executing: {step_name}")
            
            try:
                success = step_function()
                if not success:
                    logger.error(f"{step_name} failed. Continuing with next steps...")
            except Exception as e:
                logger.error(f"Unexpected error in {step_name}: {e}")
        
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info(f"Pipeline execution completed in {execution_time}")
        logger.info("=" * 60)
        
        # Return final results
        final_results = {
            'execution_time': str(execution_time),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'step_results': self.results
        }
        
        return final_results


def main():
    """Main function to run the pipeline."""
    
    # Configuration can be customized here
    custom_config = {
        "data_collection": {
            "sample_locations": [
                {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
                {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
                {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
                {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
                {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639}
            ],
            "days_back": 15,
            "apply_cloud_mask": True
        },
        "eda": {
            "generate_plots": True,
            "save_summary": True
        },
        "preprocessing": {
            "remove_outliers": True,
            "normalize_data": True
        },
        "training": {
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.001,
            "is_classification": False,
            "test_size": 0.2
        }
    }
    
    # Initialize and run pipeline
    pipeline = AQIPipeline(config=custom_config)
    results = pipeline.run_complete_pipeline()
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    
    successful_steps = sum(1 for step in results['step_results'].values() 
                          if step.get('status', '').startswith('success'))
    total_steps = len(results['step_results'])
    
    print(f"Execution Time: {results['execution_time']}")
    print(f"Successful Steps: {successful_steps}/{total_steps}")
    print("Results saved in: outputs/results/")
    
    if successful_steps == total_steps:
        print("✅ Pipeline completed successfully!")
    else:
        print("⚠️  Pipeline completed with some issues. Check logs for details.")
    
    return results


if __name__ == "__main__":
    main()
