"""
Exploratory Data Analysis utilities for AQI prediction dataset.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config.settings import VIS_CONFIG
from ..preprocessing.data_cleaner import DataCleaner

logger = logging.getLogger(__name__)


class ExploratoryAnalysis:
    """Comprehensive EDA for AQI prediction dataset."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize exploratory analysis.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = Path(output_dir) if output_dir else Path("eda_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set visualization style
        plt.style.use(VIS_CONFIG.get('style', 'default'))
        sns.set_palette(VIS_CONFIG.get('color_palette', 'viridis'))
        
        self.data_cleaner = DataCleaner()
        logger.info(f"ExploratoryAnalysis initialized with output directory: {self.output_dir}")
    
    def basic_statistics(self, df: pd.DataFrame, 
                        save_results: bool = True) -> Dict[str, Any]:
        """
        Generate basic statistics for the dataset.
        
        Args:
            df: DataFrame to analyze
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with basic statistics
        """
        logger.info("Generating basic statistics")
        
        stats = {
            'dataset_shape': df.shape,
            'column_info': {
                'total_columns': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns)
            },
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_statistics'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            stats['categorical_statistics'] = {}
            for col in categorical_cols:
                stats['categorical_statistics'][col] = {
                    'unique_values': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'frequency_counts': df[col].value_counts().head(10).to_dict()
                }
        
        if save_results:
            # Save statistics to JSON
            import json
            stats_file = self.output_dir / "basic_statistics.json"
            with open(stats_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                json.dump(stats, f, indent=2, default=convert_numpy)
            
            logger.info(f"Basic statistics saved to: {stats_file}")
        
        return stats
    
    def correlation_analysis(self, df: pd.DataFrame,
                           method: str = 'pearson',
                           save_plot: bool = True) -> pd.DataFrame:
        """
        Perform correlation analysis on numeric columns.
        
        Args:
            df: DataFrame to analyze
            method: Correlation method ('pearson', 'spearman', 'kendall')
            save_plot: Whether to save correlation heatmap
            
        Returns:
            Correlation matrix
        """
        logger.info(f"Performing correlation analysis using {method} method")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.warning("Insufficient numeric columns for correlation analysis")
            return pd.DataFrame()
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr(method=method)
        
        if save_plot:
            # Create correlation heatmap
            plt.figure(figsize=VIS_CONFIG.get('figure_size', (12, 8)))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Generate heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                       center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            
            plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_file = self.output_dir / f"correlation_matrix_{method}.png"
            plt.savefig(plot_file, dpi=VIS_CONFIG.get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation heatmap saved to: {plot_file}")
        
        return corr_matrix
    
    def aqi_distribution_analysis(self, df: pd.DataFrame,
                                aqi_col: str = 'aqi',
                                save_plots: bool = True) -> Dict[str, Any]:
        """
        Analyze AQI distribution and categories.
        
        Args:
            df: DataFrame with AQI data
            aqi_col: AQI column name
            save_plots: Whether to save plots
            
        Returns:
            Dictionary with AQI analysis results
        """
        logger.info("Analyzing AQI distribution")
        
        if aqi_col not in df.columns:
            logger.error(f"AQI column '{aqi_col}' not found")
            return {}
        
        # Basic AQI statistics
        aqi_stats = {
            'mean': df[aqi_col].mean(),
            'median': df[aqi_col].median(),
            'std': df[aqi_col].std(),
            'min': df[aqi_col].min(),
            'max': df[aqi_col].max(),
            'quartiles': df[aqi_col].quantile([0.25, 0.5, 0.75]).to_dict()
        }
        
        # Add AQI categories if not present
        if 'aqi_category' not in df.columns:
            df = self.data_cleaner.add_aqi_category(df, aqi_col)
        
        # Category distribution
        category_counts = df['aqi_category'].value_counts()
        category_percentages = df['aqi_category'].value_counts(normalize=True) * 100
        
        aqi_stats['category_distribution'] = {
            'counts': category_counts.to_dict(),
            'percentages': category_percentages.to_dict()
        }
        
        if save_plots:
            # Create subplot figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('AQI Distribution Analysis', fontsize=16, fontweight='bold')
            
            # 1. AQI histogram
            axes[0, 0].hist(df[aqi_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(aqi_stats['mean'], color='red', linestyle='--', 
                              label=f"Mean: {aqi_stats['mean']:.1f}")
            axes[0, 0].axvline(aqi_stats['median'], color='green', linestyle='--', 
                              label=f"Median: {aqi_stats['median']:.1f}")
            axes[0, 0].set_xlabel('AQI Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('AQI Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. AQI box plot
            axes[0, 1].boxplot(df[aqi_col], vert=True)
            axes[0, 1].set_ylabel('AQI Value')
            axes[0, 1].set_title('AQI Box Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Category distribution bar plot
            category_counts.plot(kind='bar', ax=axes[1, 0], color='lightcoral', alpha=0.8)
            axes[1, 0].set_xlabel('AQI Category')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('AQI Category Distribution')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Category distribution pie chart
            axes[1, 1].pie(category_counts.values, labels=category_counts.index, 
                          autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('AQI Category Proportions')
            
            plt.tight_layout()
            
            plot_file = self.output_dir / "aqi_distribution_analysis.png"
            plt.savefig(plot_file, dpi=VIS_CONFIG.get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"AQI distribution plots saved to: {plot_file}")
        
        return aqi_stats
    
    def geographical_analysis(self, df: pd.DataFrame,
                            lat_col: str = 'latitude',
                            lon_col: str = 'longitude',
                            aqi_col: str = 'aqi',
                            save_plots: bool = True) -> Dict[str, Any]:
        """
        Analyze geographical distribution of AQI values.
        
        Args:
            df: DataFrame with geographical and AQI data
            lat_col: Latitude column name
            lon_col: Longitude column name
            aqi_col: AQI column name
            save_plots: Whether to save plots
            
        Returns:
            Dictionary with geographical analysis results
        """
        logger.info("Performing geographical analysis")
        
        required_cols = [lat_col, lon_col, aqi_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Required columns missing: {missing_cols}")
            return {}
        
        # Geographical statistics
        geo_stats = {
            'coordinate_ranges': {
                'latitude': {'min': df[lat_col].min(), 'max': df[lat_col].max()},
                'longitude': {'min': df[lon_col].min(), 'max': df[lon_col].max()}
            },
            'data_coverage': {
                'total_locations': len(df),
                'unique_coordinates': df[[lat_col, lon_col]].drop_duplicates().shape[0]
            }
        }
        
        if save_plots:
            # Create geographical plots
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle('Geographical Analysis of AQI Data', fontsize=16, fontweight='bold')
            
            # 1. Scatter plot of coordinates colored by AQI
            scatter = axes[0].scatter(df[lon_col], df[lat_col], c=df[aqi_col], 
                                    cmap='RdYlBu_r', s=50, alpha=0.6)
            axes[0].set_xlabel('Longitude')
            axes[0].set_ylabel('Latitude')
            axes[0].set_title('AQI Values by Geographic Location')
            axes[0].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axes[0])
            cbar.set_label('AQI Value')
            
            # 2. Hexbin plot for density
            hexbin = axes[1].hexbin(df[lon_col], df[lat_col], C=df[aqi_col], 
                                  gridsize=30, cmap='RdYlBu_r', mincnt=1)
            axes[1].set_xlabel('Longitude')
            axes[1].set_ylabel('Latitude')
            axes[1].set_title('AQI Density Heatmap')
            
            # Add colorbar
            cbar2 = plt.colorbar(hexbin, ax=axes[1])
            cbar2.set_label('Mean AQI Value')
            
            plt.tight_layout()
            
            plot_file = self.output_dir / "geographical_analysis.png"
            plt.savefig(plot_file, dpi=VIS_CONFIG.get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Geographical analysis plots saved to: {plot_file}")
        
        return geo_stats
    
    def temporal_analysis(self, df: pd.DataFrame,
                         date_col: str,
                         aqi_col: str = 'aqi',
                         save_plots: bool = True) -> Dict[str, Any]:
        """
        Analyze temporal patterns in AQI data.
        
        Args:
            df: DataFrame with temporal and AQI data
            date_col: Date column name
            aqi_col: AQI column name
            save_plots: Whether to save plots
            
        Returns:
            Dictionary with temporal analysis results
        """
        logger.info("Performing temporal analysis")
        
        if date_col not in df.columns or aqi_col not in df.columns:
            logger.error(f"Required columns '{date_col}' or '{aqi_col}' not found")
            return {}
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract time components
        df_temp = df.copy()
        df_temp['year'] = df_temp[date_col].dt.year
        df_temp['month'] = df_temp[date_col].dt.month
        df_temp['day_of_week'] = df_temp[date_col].dt.dayofweek
        df_temp['hour'] = df_temp[date_col].dt.hour
        
        # Temporal statistics
        temporal_stats = {
            'date_range': {
                'start': df_temp[date_col].min().strftime('%Y-%m-%d'),
                'end': df_temp[date_col].max().strftime('%Y-%m-%d'),
                'span_days': (df_temp[date_col].max() - df_temp[date_col].min()).days
            },
            'yearly_trends': df_temp.groupby('year')[aqi_col].agg(['mean', 'std', 'count']).to_dict(),
            'monthly_trends': df_temp.groupby('month')[aqi_col].agg(['mean', 'std', 'count']).to_dict(),
            'daily_trends': df_temp.groupby('day_of_week')[aqi_col].agg(['mean', 'std', 'count']).to_dict()
        }
        
        if save_plots:
            # Create temporal plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Temporal Analysis of AQI Data', fontsize=16, fontweight='bold')
            
            # 1. Time series plot
            monthly_avg = df_temp.groupby(df_temp[date_col].dt.to_period('M'))[aqi_col].mean()
            axes[0, 0].plot(monthly_avg.index.to_timestamp(), monthly_avg.values, 
                           marker='o', linewidth=2)
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Average AQI')
            axes[0, 0].set_title('Monthly AQI Trends')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Monthly averages
            monthly_stats = df_temp.groupby('month')[aqi_col].mean()
            axes[0, 1].bar(monthly_stats.index, monthly_stats.values, color='lightblue', alpha=0.8)
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Average AQI')
            axes[0, 1].set_title('Average AQI by Month')
            axes[0, 1].set_xticks(range(1, 13))
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Day of week patterns
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_stats = df_temp.groupby('day_of_week')[aqi_col].mean()
            axes[1, 0].bar(range(7), dow_stats.values, color='lightgreen', alpha=0.8)
            axes[1, 0].set_xlabel('Day of Week')
            axes[1, 0].set_ylabel('Average AQI')
            axes[1, 0].set_title('Average AQI by Day of Week')
            axes[1, 0].set_xticks(range(7))
            axes[1, 0].set_xticklabels(day_names)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Hourly patterns (if hour data available)
            if 'hour' in df_temp.columns and df_temp['hour'].notna().any():
                hourly_stats = df_temp.groupby('hour')[aqi_col].mean()
                axes[1, 1].plot(hourly_stats.index, hourly_stats.values, 
                               marker='o', color='red', linewidth=2)
                axes[1, 1].set_xlabel('Hour of Day')
                axes[1, 1].set_ylabel('Average AQI')
                axes[1, 1].set_title('Average AQI by Hour')
                axes[1, 1].set_xticks(range(0, 24, 4))
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Hourly data not available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Hourly Analysis (No Data)')
            
            plt.tight_layout()
            
            plot_file = self.output_dir / "temporal_analysis.png"
            plt.savefig(plot_file, dpi=VIS_CONFIG.get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Temporal analysis plots saved to: {plot_file}")
        
        return temporal_stats
    
    def comprehensive_eda(self, df: pd.DataFrame,
                         lat_col: str = 'latitude',
                         lon_col: str = 'longitude',
                         aqi_col: str = 'aqi',
                         date_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory data analysis.
        
        Args:
            df: DataFrame to analyze
            lat_col: Latitude column name
            lon_col: Longitude column name
            aqi_col: AQI column name
            date_col: Date column name (optional)
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting comprehensive EDA")
        
        eda_results = {
            'basic_statistics': self.basic_statistics(df),
            'correlation_analysis': self.correlation_analysis(df).to_dict(),
            'aqi_distribution': self.aqi_distribution_analysis(df, aqi_col),
            'geographical_analysis': self.geographical_analysis(df, lat_col, lon_col, aqi_col)
        }
        
        # Add temporal analysis if date column provided
        if date_col and date_col in df.columns:
            eda_results['temporal_analysis'] = self.temporal_analysis(df, date_col, aqi_col)
        
        # Save comprehensive report
        import json
        report_file = self.output_dir / "comprehensive_eda_report.json"
        with open(report_file, 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                return obj
            
            json.dump(eda_results, f, indent=2, default=convert_numpy)
        
        logger.info(f"Comprehensive EDA report saved to: {report_file}")
        logger.info("Comprehensive EDA completed")
        
        return eda_results
