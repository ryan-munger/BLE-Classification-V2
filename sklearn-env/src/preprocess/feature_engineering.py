import pandas as pd
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

def calculate_frequency(group):
    """
    Calculate advertisement frequency in a memory-efficient way.
    """
    if len(group) == 0:
        return pd.Series()
    
    # Calculate total time span in seconds
    min_time = group['Timestamp'].min()
    max_time = group['Timestamp'].max()
    time_span = max_time - min_time
    
    if time_span == 0:
        return pd.Series([len(group)], index=[0])
    
    # Calculate frequency (packets per second)
    frequency = len(group) / time_span
    return pd.Series([frequency], index=[0])

def feature_engineering(cleaned_csv_path, output_directory):
    """
    Performs feature engineering on the cleaned BLE dataset in a memory-efficient way.

    Args:
        cleaned_csv_path (str): Path to the cleaned CSV file.
        output_directory (str): Path to save the feature engineered CSV.
    """
    print("\n=== Starting Feature Engineering Process ===")
    print(f"Reading CSV from: {cleaned_csv_path}")
    
    try:
        # Read data in chunks
        chunk_size = 100000
        chunks = pd.read_csv(cleaned_csv_path, chunksize=chunk_size)
        print("Processing data in chunks...")
        
        # Initialize lists to store results
        all_inter_arrival_times = []
        all_frequencies = []
        all_sources = []
        
        # Process each chunk
        for chunk in tqdm(chunks, desc="Processing chunks"):
            # Calculate inter-arrival times
            groups = chunk.groupby('Source')
            for source, group in groups:
                # Sort by timestamp
                group = group.sort_values('Timestamp')
                
                # Calculate inter-arrival times
                timestamps = group['Timestamp'].values
                if len(timestamps) > 1:
                    diffs = np.diff(timestamps)
                    all_inter_arrival_times.extend(diffs)
                    all_sources.extend([source] * len(diffs))
                
                # Calculate frequency
                freq = calculate_frequency(group)
                if not freq.empty:
                    all_frequencies.append((source, freq.iloc[0]))
        
        # Create final DataFrame
        print("\nCreating final feature engineered dataset...")
        df = pd.read_csv(cleaned_csv_path)
        
        # Add inter-arrival times
        inter_arrival_df = pd.DataFrame({
            'Source': all_sources,
            'Inter-arrival Time': all_inter_arrival_times
        })
        
        # Add frequencies
        freq_df = pd.DataFrame(all_frequencies, columns=['Source', 'Frequency'])
        
        # Merge features back to original data
        df = df.merge(inter_arrival_df, on='Source', how='left')
        df = df.merge(freq_df, on='Source', how='left')
        
        # Fill NaN values
        df['Inter-arrival Time'] = df['Inter-arrival Time'].fillna(0)
        df['Frequency'] = df['Frequency'].fillna(0)
        
        # Save the feature engineered dataset
        output_path = os.path.join(output_directory, 'feature_engineered_dataset.csv')
        print(f"Saving feature engineered dataset to {output_path}...")
        df.to_csv(output_path, index=False)
        
        return df
        
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        sys.exit(1)

def main():
    """
    Main function to execute the feature engineering process.
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