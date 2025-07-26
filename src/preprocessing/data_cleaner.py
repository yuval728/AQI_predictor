"""
Data cleaning and validation utilities for AQI prediction dataset.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple

from config.settings import AQI_CATEGORIES

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning and validation for AQI prediction."""
    
    def __init__(self):
        """Initialize data cleaner."""
        self.aqi_categories = AQI_CATEGORIES
        logger.info("DataCleaner initialized")
    
    def validate_coordinates(self, df: pd.DataFrame, 
                           lat_col: str = 'latitude', 
                           lon_col: str = 'longitude') -> pd.DataFrame:
        """
        Validate and clean coordinate data.
        
        Args:
            df: DataFrame with coordinate data
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Validating coordinates")
        initial_count = len(df)
        
        # Check if columns exist
        if lat_col not in df.columns or lon_col not in df.columns:
            logger.error(f"Required columns {lat_col} and/or {lon_col} not found")
            return df
        
        # Remove rows with missing coordinates
        df = df.dropna(subset=[lat_col, lon_col])
        
        # Validate latitude range [-90, 90]
        df = df[(df[lat_col] >= -90) & (df[lat_col] <= 90)]
        
        # Validate longitude range [-180, 180]
        df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180)]
        
        # Remove duplicate coordinates
        df = df.drop_duplicates(subset=[lat_col, lon_col])
        
        final_count = len(df)
        logger.info(f"Coordinate validation: {initial_count} -> {final_count} rows "
                   f"({initial_count - final_count} rows removed)")
        
        return df
    
    def validate_aqi_values(self, df: pd.DataFrame, 
                          aqi_col: str = 'aqi') -> pd.DataFrame:
        """
        Validate and clean AQI values.
        
        Args:
            df: DataFrame with AQI data
            aqi_col: Name of AQI column
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Validating AQI values")
        initial_count = len(df)
        
        if aqi_col not in df.columns:
            logger.error(f"AQI column '{aqi_col}' not found")
            return df
        
        # Remove rows with missing AQI values
        df = df.dropna(subset=[aqi_col])
        
        # Convert to numeric, coercing errors to NaN
        df[aqi_col] = pd.to_numeric(df[aqi_col], errors='coerce')
        
        # Remove rows where conversion failed
        df = df.dropna(subset=[aqi_col])
        
        # Validate AQI range [0, 500]
        df = df[(df[aqi_col] >= 0) & (df[aqi_col] <= 500)]
        
        final_count = len(df)
        logger.info(f"AQI validation: {initial_count} -> {final_count} rows "
                   f"({initial_count - final_count} rows removed)")
        
        return df
    
    def add_aqi_category(self, df: pd.DataFrame, 
                        aqi_col: str = 'aqi',
                        category_col: str = 'aqi_category') -> pd.DataFrame:
        """
        Add AQI category column based on AQI values.
        
        Args:
            df: DataFrame with AQI data
            aqi_col: Name of AQI column
            category_col: Name of new category column
            
        Returns:
            DataFrame with AQI category column
        """
        logger.info("Adding AQI categories")
        
        if aqi_col not in df.columns:
            logger.error(f"AQI column '{aqi_col}' not found")
            return df
        
        def categorize_aqi(aqi_value):
            for category, (min_val, max_val) in self.aqi_categories.items():
                if min_val <= aqi_value <= max_val:
                    return category
            return "Unknown"
        
        df[category_col] = df[aqi_col].apply(categorize_aqi)
        
        # Log category distribution
        category_counts = df[category_col].value_counts()
        logger.info(f"AQI category distribution:\n{category_counts}")
        
        return df
    
    def clean_aqi_values(self, df: pd.DataFrame, aqi_col: str = 'aqi') -> pd.DataFrame:
        """
        Alias for validate_aqi_values method.
        
        Args:
            df: DataFrame with AQI data
            aqi_col: Name of AQI column
            
        Returns:
            Cleaned DataFrame
        """
        return self.validate_aqi_values(df, aqi_col)
    
    def create_aqi_categories(self, df: pd.DataFrame, aqi_col: str = 'aqi') -> pd.DataFrame:
        """
        Alias for add_aqi_category method.
        
        Args:
            df: DataFrame with AQI data
            aqi_col: Name of AQI column
            
        Returns:
            DataFrame with AQI category column
        """
        return self.add_aqi_category(df, aqi_col)
    
    def normalize_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Normalize numerical features using Min-Max scaling.
        
        Args:
            df: DataFrame with features to normalize
            features: List of feature column names
            
        Returns:
            DataFrame with normalized features
        """
        logger.info(f"Normalizing features: {features}")
        
        from sklearn.preprocessing import MinMaxScaler
        
        df_normalized = df.copy()
        scaler = MinMaxScaler()
        
        for feature in features:
            if feature in df_normalized.columns:
                df_normalized[f'{feature}_normalized'] = scaler.fit_transform(
                    df_normalized[[feature]]
                ).flatten()
        
        return df_normalized
    
    def remove_outliers(self, df: pd.DataFrame, 
                       columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from specified columns.
        
        Args:
            df: DataFrame
            columns: List of column names to check for outliers
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"Removing outliers using {method} method")
        initial_count = len(df)
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
            
            if method == 'iqr':
                # Interquartile Range method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                before_count = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                after_count = len(df)
                
                logger.info(f"Column '{col}': removed {before_count - after_count} outliers "
                           f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})")
            
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                before_count = len(df)
                df = df[z_scores <= threshold]
                after_count = len(df)
                
                logger.info(f"Column '{col}': removed {before_count - after_count} outliers "
                           f"(z-score threshold: {threshold})")
        
        final_count = len(df)
        logger.info(f"Outlier removal: {initial_count} -> {final_count} rows "
                   f"({initial_count - final_count} total rows removed)")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame,
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: DataFrame with missing values
            strategy: Dictionary mapping column names to strategies
                     ('drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill')
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values")
        
        if strategy is None:
            strategy = {}
        
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Missing values found in columns:\n{missing_cols}")
        
        for col in missing_cols.index:
            col_strategy = strategy.get(col, 'drop')
            
            if col_strategy == 'drop':
                df = df.dropna(subset=[col])
            elif col_strategy == 'mean' and df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
            elif col_strategy == 'median' and df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            elif col_strategy == 'mode':
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
            elif col_strategy == 'forward_fill':
                df[col].fillna(method='ffill', inplace=True)
            elif col_strategy == 'backward_fill':
                df[col].fillna(method='bfill', inplace=True)
            
            logger.info(f"Applied '{col_strategy}' strategy to column '{col}'")
        
        return df
    
    def validate_date_columns(self, df: pd.DataFrame,
                            date_columns: List[str],
                            date_format: Optional[str] = None) -> pd.DataFrame:
        """
        Validate and clean date columns.
        
        Args:
            df: DataFrame with date columns
            date_columns: List of date column names
            date_format: Expected date format (if None, will infer)
            
        Returns:
            DataFrame with validated date columns
        """
        logger.info("Validating date columns")
        
        for col in date_columns:
            if col not in df.columns:
                logger.warning(f"Date column '{col}' not found, skipping")
                continue
            
            try:
                # Convert to datetime
                if date_format:
                    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                else:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Remove rows with invalid dates
                initial_count = len(df)
                df = df.dropna(subset=[col])
                final_count = len(df)
                
                logger.info(f"Date column '{col}': {initial_count - final_count} invalid dates removed")
                
            except Exception as e:
                logger.error(f"Error processing date column '{col}': {e}")
        
        return df
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        logger.info("Generating data quality report")
        
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).to_dict()
        }
        
        # Add numerical column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report['numeric_statistics'] = df[numeric_cols].describe().to_dict()
        
        # Add categorical column statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            report['categorical_statistics'] = {}
            for col in categorical_cols:
                report['categorical_statistics'][col] = {
                    'unique_values': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
                }
        
        logger.info("Data quality report generated")
        return report
    
    def clean_dataset(self, df: pd.DataFrame,
                     lat_col: str = 'latitude',
                     lon_col: str = 'longitude',
                     aqi_col: str = 'aqi',
                     remove_outliers: bool = True,
                     missing_value_strategy: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive dataset cleaning pipeline.
        
        Args:
            df: Raw DataFrame
            lat_col: Latitude column name
            lon_col: Longitude column name
            aqi_col: AQI column name
            remove_outliers: Whether to remove outliers
            missing_value_strategy: Strategy for handling missing values
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        logger.info("Starting comprehensive dataset cleaning")
        initial_count = len(df)
        
        # Generate initial quality report
        initial_report = self.generate_data_quality_report(df)
        
        # Clean coordinates
        df = self.validate_coordinates(df, lat_col, lon_col)
        
        # Clean AQI values
        df = self.validate_aqi_values(df, aqi_col)
        
        # Add AQI categories
        df = self.add_aqi_category(df, aqi_col)
        
        # Handle missing values
        if missing_value_strategy:
            df = self.handle_missing_values(df, missing_value_strategy)
        
        # Remove outliers
        if remove_outliers:
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                          if col not in [lat_col, lon_col]]
            if numeric_cols:
                df = self.remove_outliers(df, numeric_cols)
        
        # Generate final quality report
        final_report = self.generate_data_quality_report(df)
        
        # Create cleaning summary
        cleaning_report = {
            'initial_rows': initial_count,
            'final_rows': len(df),
            'rows_removed': initial_count - len(df),
            'removal_percentage': ((initial_count - len(df)) / initial_count) * 100,
            'initial_quality': initial_report,
            'final_quality': final_report
        }
        
        logger.info(f"Dataset cleaning completed: {initial_count} -> {len(df)} rows "
                   f"({cleaning_report['removal_percentage']:.1f}% removed)")
        
        return df, cleaning_report
