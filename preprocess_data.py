"""
Data preprocessing script to prepare dataset for training.
"""
import argparse
import os
from src.data_processing import process_dataset, impute_missing_values
from src.satellite_utils import SatelliteImageFetcher


def main():
    parser = argparse.ArgumentParser(description='Preprocess AQI Dataset')
    parser.add_argument('--input_csv', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Path to save processed CSV file')
    parser.add_argument('--download_satellite', action='store_true',
                       help='Download satellite images')
    parser.add_argument('--satellite_output_dir', type=str, default='satellite_images',
                       help='Directory to save satellite images')
    parser.add_argument('--impute_missing', action='store_true',
                       help='Impute missing values using KNN')
    
    args = parser.parse_args()
    
    print("Starting data preprocessing...")
    
    # Process dataset
    print(f"Loading and processing dataset from {args.input_csv}...")
    processed_data = process_dataset(args.input_csv)
    print(f"Processed {len(processed_data)} samples")
    
    # Impute missing values if requested
    if args.impute_missing:
        print("Imputing missing values...")
        processed_data = impute_missing_values(processed_data, strategy='knn')
        print("Missing value imputation completed")
    
    # Download satellite images if requested
    if args.download_satellite:
        print("Downloading satellite images...")
        
        if not os.path.exists(args.satellite_output_dir):
            os.makedirs(args.satellite_output_dir)
        
        fetcher = SatelliteImageFetcher()
        
        # Prepare coordinates list
        coordinates_list = []
        for idx, row in processed_data.iterrows():
            if row['Latitude'] != 0 and row['Longitude'] != 0:
                # Create date string from the data
                try:
                    date_str = f"{int(row['Year'])}-{int(row['Month']):02d}-{int(row['Day']):02d}"
                    coordinates_list.append((
                        row['Normalized_Filename'], 
                        row['Latitude'], 
                        row['Longitude'], 
                        date_str
                    ))
                except Exception:
                    # Skip if date conversion fails
                    continue
        
        print(f"Downloading satellite images for {len(coordinates_list)} locations...")
        results = fetcher.batch_fetch_images(coordinates_list, args.satellite_output_dir)
        
        # Update processed data with satellite image availability
        processed_data['satellite_image_available'] = False
        for idx, row in processed_data.iterrows():
            filename = row['Normalized_Filename']
            if filename in results and results[filename]['success']:
                processed_data.at[idx, 'satellite_image_available'] = True
        
        successful_downloads = sum(1 for r in results.values() if r.get('success', False))
        print(f"Successfully downloaded {successful_downloads}/{len(coordinates_list)} satellite images")
    
    # Save processed dataset
    print(f"Saving processed dataset to {args.output_csv}...")
    processed_data.to_csv(args.output_csv, index=False)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total samples: {len(processed_data)}")
    print(f"Samples with coordinates: {len(processed_data[processed_data['Latitude'] != 0])}")
    
    if 'satellite_image_available' in processed_data.columns:
        available_satellite = len(processed_data[processed_data['satellite_image_available']])
        print(f"Samples with satellite images: {available_satellite}")
    
    print(f"Unique locations: {processed_data['Location'].nunique()}")
    print(f"Date range: {processed_data['Year'].min()}-{processed_data['Year'].max()}")
    
    # Print AQI distribution
    print("\nAQI Distribution:")
    print(processed_data['AQI'].describe())
    
    if 'AQI_Class' in processed_data.columns:
        print("\nAQI Class Distribution:")
        print(processed_data['AQI_Class'].value_counts())
    
    print("Data preprocessing completed!")


if __name__ == "__main__":
    main()
