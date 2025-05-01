import pandas as pd
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

def calculate_frequency(group):
    """
    Calculate advertisement frequency in a memory-efficient way.
    
    This function computes the frequency of BLE advertisements by:
    1. Finding the time span between first and last advertisement
    2. Calculating packets per second (frequency)
    
    Args:
        group (pd.DataFrame): A group of BLE advertisements from the same source
        
    Returns:
        float: The frequency of advertisements in packets per second
    """
    if len(group) == 0:
        return 0.0
    
    # Calculate total time span in seconds
    min_time = group['Timestamp'].min()
    max_time = group['Timestamp'].max()
    time_span = max_time - min_time
    
    if time_span == 0:
        return float(len(group))
    
    # Calculate frequency (packets per second)
    return float(len(group)) / time_span

def feature_engineering(cleaned_csv_path, output_directory):
    """
    Performs feature engineering on the cleaned BLE dataset in a memory-efficient way.
    
    This function:
    1. Processes data in chunks to handle large datasets
    2. Calculates temporal features (inter-arrival times and frequency)
    3. Preserves important BLE characteristics
    4. Optimizes memory usage through chunking and selective column retention
    
    Args:
        cleaned_csv_path (str): Path to the cleaned CSV file
        output_directory (str): Path to save the feature engineered CSV
        
    Returns:
        str: Path to the generated feature-engineered CSV file
    """
    print("\n=== Starting Feature Engineering Process ===")
    print(f"Reading CSV from: {cleaned_csv_path}")
    
    try:
        # Read data in smaller chunks to manage memory usage
        chunk_size = 10000  # Reduced chunk size for better memory management
        chunks = pd.read_csv(cleaned_csv_path, chunksize=chunk_size, low_memory=False)
        print("Processing data in chunks...")
        
        # Initialize output file
        output_path = os.path.join(output_directory, 'feature_engineered_dataset.csv')
        first_chunk = True
        
        # Process each chunk
        for chunk in tqdm(chunks, desc="Processing chunks"):
            # Ensure numeric columns are properly typed for calculations
            numeric_columns = ['Timestamp', 'RSSI', 'Channel Index', 'Packet counter', 'Power Level (dBm)']
            for col in numeric_columns:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            # Calculate features for each source (BLE device)
            sources = chunk['Source'].unique()
            chunk_features = []
            
            for source in sources:
                # Get data for this source and sort by timestamp
                source_data = chunk[chunk['Source'] == source].sort_values('Timestamp')
                
                if len(source_data) > 1:
                    # Calculate inter-arrival times between consecutive packets
                    timestamps = source_data['Timestamp'].values
                    diffs = np.diff(timestamps)
                    
                    # Calculate advertisement frequency
                    freq = calculate_frequency(source_data)
                    
                    # Create features for each row
                    for i, diff in enumerate(diffs):
                        chunk_features.append({
                            'Source': source,
                            'Inter-arrival Time': float(diff),  # Time between consecutive packets
                            'Frequency': freq  # Packets per second
                        })
            
            if chunk_features:
                # Convert features to DataFrame and merge with original chunk
                features_df = pd.DataFrame(chunk_features)
                chunk = chunk.merge(features_df, on='Source', how='left')
                
                # Fill NaN values with 0 for missing features
                chunk['Inter-arrival Time'] = chunk['Inter-arrival Time'].fillna(0)
                chunk['Frequency'] = chunk['Frequency'].fillna(0)
                
                # Select only necessary columns to reduce memory usage
                required_columns = ['Source', 'Timestamp', 'RSSI', 'Channel Index', 
                                  'Advertising Address', 'Packet counter', 
                                  'Protocol version', 'Power Level (dBm)',
                                  'Inter-arrival Time', 'Frequency', 'Label']
                chunk = chunk[required_columns]
                
                # Save chunk to CSV
                chunk.to_csv(output_path, mode='w' if first_chunk else 'a', 
                           header=first_chunk, index=False)
                first_chunk = False
            
            # Clear memory after processing each chunk
            del chunk
            del chunk_features
            del features_df
        
        print(f"\nFeature engineering completed. Dataset saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        sys.exit(1)

def main():
    """
    Main function to execute the feature engineering process.
    
    Handles command-line arguments and orchestrates the feature engineering workflow.
    """
    parser = argparse.ArgumentParser(description="Perform feature engineering on cleaned BLE data.")
    parser.add_argument('--cleaned_csv', required=True, help='Path to the cleaned CSV file.')
    parser.add_argument('--output_dir', required=True, help='Path to the directory to save the feature engineered CSV.')
    args = parser.parse_args()

    cleaned_csv_path = args.cleaned_csv
    output_directory = args.output_dir

    if not os.path.exists(cleaned_csv_path):
        print(f"Error: Cleaned CSV file not found at {cleaned_csv_path}")
        sys.exit(1)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    feature_engineered_csv_path = feature_engineering(cleaned_csv_path, output_directory)
    return feature_engineered_csv_path

if __name__ == "__main__":
    main() 