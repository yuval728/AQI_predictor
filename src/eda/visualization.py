"""
Visualization utilities for AQI prediction project.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from typing import Optional, List
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import VIS_CONFIG

logger = logging.getLogger(__name__)


class AQIVisualizer:
    """Advanced visualization tools for AQI data analysis."""
    
    def __init__(self, output_dir: Optional[str] = None, style: Optional[str] = None):
        """
        Initialize AQI visualizer.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir) if output_dir else Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plot_style = style or VIS_CONFIG.get('style', 'seaborn-v0_8')
        try:
            plt.style.use(plot_style)
        except OSError:
            plt.style.use('default')
            logger.warning(f"Style '{plot_style}' not found, using default")
        
        # Set color palette
        sns.set_palette(VIS_CONFIG.get('color_palette', 'viridis'))
        
        # Default figure parameters
        self.figsize = VIS_CONFIG.get('figure_size', (12, 8))
        self.dpi = VIS_CONFIG.get('dpi', 300)
        
        logger.info(f"AQIVisualizer initialized with output directory: {self.output_dir}")
    
    def plot_aqi_time_series(self, df: pd.DataFrame,
                           date_col: str,
                           aqi_col: str = 'aqi',
                           location_col: Optional[str] = None,
                           interactive: bool = False,
                           save_plot: bool = True) -> Optional[str]:
        """
        Create AQI time series plot.
        
        Args:
            df: DataFrame with time series data
            date_col: Date column name
            aqi_col: AQI column name
            location_col: Location column for grouping (optional)
            interactive: Whether to create interactive plot
            save_plot: Whether to save plot
            
        Returns:
            Path to saved plot file
        """
        logger.info("Creating AQI time series plot")
        
        # Ensure date column is datetime
        df_plot = df.copy()
        df_plot[date_col] = pd.to_datetime(df_plot[date_col])
        
        if interactive:
            # Create interactive plotly plot
            if location_col and location_col in df_plot.columns:
                fig = px.line(df_plot, x=date_col, y=aqi_col, color=location_col,
                             title='AQI Time Series by Location',
                             labels={date_col: 'Date', aqi_col: 'AQI Value'})
            else:
                fig = px.line(df_plot, x=date_col, y=aqi_col,
                             title='AQI Time Series',
                             labels={date_col: 'Date', aqi_col: 'AQI Value'})
            
            # Add AQI threshold lines
            thresholds = [50, 100, 150, 200, 300]
            colors = ['green', 'yellow', 'orange', 'red', 'purple']
            names = ['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy']
            
            for threshold, color, name in zip(thresholds, colors, names):
                fig.add_hline(y=threshold, line_dash="dash", line_color=color,
                             annotation_text=f"{name} ({threshold})")
            
            if save_plot:
                plot_file = self.output_dir / "aqi_time_series_interactive.html"
                fig.write_html(plot_file)
                logger.info(f"Interactive time series plot saved to: {plot_file}")
                return str(plot_file)
        
        else:
            # Create matplotlib plot
            plt.figure(figsize=self.figsize)
            
            if location_col and location_col in df_plot.columns:
                # Plot by location
                for location in df_plot[location_col].unique()[:10]:  # Limit to 10 locations
                    location_data = df_plot[df_plot[location_col] == location]
                    plt.plot(location_data[date_col], location_data[aqi_col], 
                           label=location, alpha=0.7, linewidth=2)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Single time series
                plt.plot(df_plot[date_col], df_plot[aqi_col], 
                        linewidth=2, color='blue', alpha=0.8)
            
            # Add AQI threshold lines
            thresholds = [50, 100, 150, 200, 300]
            colors = ['green', 'yellow', 'orange', 'red', 'purple']
            names = ['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy']
            
            for threshold, color, name in zip(thresholds, colors, names):
                plt.axhline(y=threshold, color=color, linestyle='--', alpha=0.7,
                           label=f"{name} ({threshold})")
            
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('AQI Value', fontsize=12)
            plt.title('AQI Time Series', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_plot:
                plot_file = self.output_dir / "aqi_time_series.png"
                plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Time series plot saved to: {plot_file}")
                return str(plot_file)
    
    def plot_geographical_heatmap(self, df: pd.DataFrame,
                                lat_col: str = 'latitude',
                                lon_col: str = 'longitude',
                                aqi_col: str = 'aqi',
                                interactive: bool = True,
                                save_plot: bool = True) -> Optional[str]:
        """
        Create geographical heatmap of AQI values.
        
        Args:
            df: DataFrame with geographical and AQI data
            lat_col: Latitude column name
            lon_col: Longitude column name
            aqi_col: AQI column name
            interactive: Whether to create interactive map
            save_plot: Whether to save plot
            
        Returns:
            Path to saved plot file
        """
        logger.info("Creating geographical AQI heatmap")
        
        if interactive:
            # Create interactive plotly map
            fig = px.density_mapbox(df, lat=lat_col, lon=lon_col, z=aqi_col,
                                  radius=10, center=dict(lat=df[lat_col].mean(), 
                                                        lon=df[lon_col].mean()),
                                  zoom=5, mapbox_style="open-street-map",
                                  title="AQI Geographical Distribution",
                                  color_continuous_scale="RdYlBu_r")
            
            if save_plot:
                plot_file = self.output_dir / "geographical_heatmap_interactive.html"
                fig.write_html(plot_file)
                logger.info(f"Interactive geographical heatmap saved to: {plot_file}")
                return str(plot_file)
        
        else:
            # Create matplotlib plot
            plt.figure(figsize=self.figsize)
            
            # Create scatter plot with color mapping
            scatter = plt.scatter(df[lon_col], df[lat_col], c=df[aqi_col], 
                                cmap='RdYlBu_r', s=50, alpha=0.6)
            
            plt.colorbar(scatter, label='AQI Value')
            plt.xlabel('Longitude', fontsize=12)
            plt.ylabel('Latitude', fontsize=12)
            plt.title('AQI Geographical Distribution', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_plot:
                plot_file = self.output_dir / "geographical_heatmap.png"
                plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                logger.info(f"Geographical heatmap saved to: {plot_file}")
                return str(plot_file)
    
    def plot_aqi_categories_distribution(self, df: pd.DataFrame,
                                       category_col: str = 'aqi_category',
                                       save_plot: bool = True) -> Optional[str]:
        """
        Create AQI categories distribution plots.
        
        Args:
            df: DataFrame with AQI category data
            category_col: AQI category column name
            save_plot: Whether to save plot
            
        Returns:
            Path to saved plot file
        """
        logger.info("Creating AQI categories distribution plot")
        
        if category_col not in df.columns:
            logger.error(f"Category column '{category_col}' not found")
            return None
        
        # Get category counts
        category_counts = df[category_col].value_counts()
        
        # Create subplot figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('AQI Categories Distribution', fontsize=16, fontweight='bold')
        
        # Bar plot
        category_counts.plot(kind='bar', ax=axes[0], color='lightcoral', alpha=0.8)
        axes[0].set_xlabel('AQI Category')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Category Counts')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(category_counts.values):
            axes[0].text(i, v + max(category_counts.values) * 0.01, str(v), 
                        ha='center', va='bottom')
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
        wedges, texts, autotexts = axes[1].pie(category_counts.values, 
                                             labels=category_counts.index,
                                             autopct='%1.1f%%', 
                                             startangle=90,
                                             colors=colors)
        axes[1].set_title('Category Proportions')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.output_dir / "aqi_categories_distribution.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"AQI categories distribution plot saved to: {plot_file}")
            return str(plot_file)
    
    def plot_correlation_matrix(self, df: pd.DataFrame,
                              method: str = 'pearson',
                              save_plot: bool = True) -> Optional[str]:
        """
        Create enhanced correlation matrix heatmap.
        
        Args:
            df: DataFrame with numeric data
            method: Correlation method
            save_plot: Whether to save plot
            
        Returns:
            Path to saved plot file
        """
        logger.info(f"Creating correlation matrix heatmap using {method} method")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.warning("Insufficient numeric columns for correlation analysis")
            return None
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr(method=method)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        # Generate custom colormap
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, annot=True, fmt='.2f', 
                   cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                   linewidths=0.5)
        
        plt.title(f'Feature Correlation Matrix ({method.capitalize()})', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.output_dir / f"correlation_matrix_{method}.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"Correlation matrix saved to: {plot_file}")
            return str(plot_file)
    
    def plot_feature_importance(self, feature_names: List[str],
                              importance_values: List[float],
                              title: str = 'Feature Importance',
                              save_plot: bool = True) -> Optional[str]:
        """
        Create feature importance plot.
        
        Args:
            feature_names: List of feature names
            importance_values: List of importance values
            title: Plot title
            save_plot: Whether to save plot
            
        Returns:
            Path to saved plot file
        """
        logger.info("Creating feature importance plot")
        
        # Create DataFrame for easier handling
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
        
        plt.barh(importance_df['feature'], importance_df['importance'], 
                color='skyblue', alpha=0.8)
        
        # Add value labels on bars
        for i, (feature, importance) in enumerate(zip(importance_df['feature'], 
                                                     importance_df['importance'])):
            plt.text(importance + max(importance_values) * 0.01, i, f'{importance:.3f}',
                    va='center', ha='left', fontweight='bold')
        
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.output_dir / "feature_importance.png"
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"Feature importance plot saved to: {plot_file}")
            return str(plot_file)
    
    def create_dashboard(self, df: pd.DataFrame,
                        lat_col: str = 'latitude',
                        lon_col: str = 'longitude',
                        aqi_col: str = 'aqi',
                        date_col: Optional[str] = None,
                        save_dashboard: bool = True) -> Optional[str]:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            df: DataFrame with AQI data
            lat_col: Latitude column name
            lon_col: Longitude column name
            aqi_col: AQI column name
            date_col: Date column name (optional)
            save_dashboard: Whether to save dashboard
            
        Returns:
            Path to saved dashboard file
        """
        logger.info("Creating comprehensive AQI dashboard")
        
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('AQI Distribution', 'Geographical Distribution',
                          'AQI Categories', 'Correlation Heatmap',
                          'Time Series', 'Statistics Summary'),
            specs=[[{'type': 'histogram'}, {'type': 'mapbox'}],
                   [{'type': 'bar'}, {'type': 'heatmap'}],
                   [{'type': 'scatter'}, {'type': 'table'}]]
        )
        
        # 1. AQI Distribution
        fig.add_trace(
            go.Histogram(x=df[aqi_col], name='AQI Distribution', 
                        marker_color='skyblue', opacity=0.7),
            row=1, col=1
        )
        
        # 2. Geographical Distribution
        if len(df) > 0:
            fig.add_trace(
                go.Scattermapbox(
                    lat=df[lat_col], lon=df[lon_col],
                    mode='markers',
                    marker=dict(size=8, color=df[aqi_col], 
                              colorscale='RdYlBu_r', showscale=True),
                    text=df[aqi_col],
                    name='AQI Values'
                ),
                row=1, col=2
            )
        
        # 3. AQI Categories (if available)
        if 'aqi_category' in df.columns:
            category_counts = df['aqi_category'].value_counts()
            fig.add_trace(
                go.Bar(x=category_counts.index, y=category_counts.values,
                      name='Category Counts', marker_color='lightcoral'),
                row=2, col=1
            )
        
        # 4. Correlation Heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns,
                          y=corr_matrix.columns, colorscale='RdBu',
                          name='Correlation'),
                row=2, col=2
            )
        
        # 5. Time Series (if date column available)
        if date_col and date_col in df.columns:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            daily_avg = df_temp.groupby(df_temp[date_col].dt.date)[aqi_col].mean().reset_index()
            
            fig.add_trace(
                go.Scatter(x=daily_avg[date_col], y=daily_avg[aqi_col],
                          mode='lines+markers', name='Daily Average AQI'),
                row=3, col=1
            )
        
        # 6. Statistics Summary Table
        stats_data = [
            ['Mean AQI', f"{df[aqi_col].mean():.2f}"],
            ['Median AQI', f"{df[aqi_col].median():.2f}"],
            ['Std Dev', f"{df[aqi_col].std():.2f}"],
            ['Min AQI', f"{df[aqi_col].min():.2f}"],
            ['Max AQI', f"{df[aqi_col].max():.2f}"],
            ['Total Records', f"{len(df)}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=list(zip(*stats_data)),
                          fill_color='lavender',
                          align='left')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="AQI Analysis Dashboard",
            title_x=0.5,
            height=1200,
            showlegend=False
        )
        
        # Update mapbox
        if len(df) > 0:
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=df[lat_col].mean(), lon=df[lon_col].mean()),
                    zoom=5
                )
            )
        
        if save_dashboard:
            dashboard_file = self.output_dir / "aqi_dashboard.html"
            fig.write_html(dashboard_file)
            logger.info(f"Interactive dashboard saved to: {dashboard_file}")
            return str(dashboard_file)
        
        return None
