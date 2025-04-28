import pandas as pd
import numpy as np
import os
import sys
import argparse

def feature_engineering(cleaned_csv_path, output_directory):
    """
    Performs feature engineering on the cleaned BLE dataset.

    Args:
        cleaned_csv_path (str): Path to the cleaned CSV file.
        output_directory (str): Path to save the feature engineered CSV.
    """
    print(f"Starting feature engineering process...")
    print(f"Reading CSV from: {cleaned_csv_path}")
    try:
        df = pd.read_csv(cleaned_csv_path)
        print(f"Successfully read CSV with {len(df)} rows")
    except Exception as e:
        print(f"Error reading cleaned CSV: {e}")
        sys.exit(1)

    print(f"Feature engineering on: {cleaned_csv_path}")
    print("Calculating inter-arrival times...")

    # --- Inter-arrival Time ---
    def calculate_inter_arrival_time(group):
        """Calculates the time difference between consecutive packets for each source."""
        timestamps = group['Timestamp'].values
        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            return pd.Series(diffs, index=group.index[1:])
        return pd.Series(index=group.index)

    # Calculate inter-arrival time for each source
    groups = df.groupby('Source')
    inter_arrival_times = []
    
    for _, group in groups:
        times = calculate_inter_arrival_time(group)
        inter_arrival_times.append(times)
    
    df['Inter-arrival Time'] = pd.concat(inter_arrival_times).reindex(df.index).fillna(0)

    # --- Advertisement Frequency (Packets per Second) ---
    def calculate_frequency(group):
        """Calculates the number of advertisements per second for each source."""
        # Window size of 1 second.  Can be adjusted.
        bins = pd.cut(group['Timestamp'], bins=np.arange(group['Timestamp'].min(), group['Timestamp'].max() + 1))
        return bins.value_counts().reindex(bins.unique()).fillna(0)

    # Calculate frequency for each source
    frequencies = []
    for _, group in df.groupby('Source'):
        freq = calculate_frequency(group)
        freq_df = pd.DataFrame({'Source': group['Source'].iloc[0], 'Timestamp_bin': freq.index, 'Frequency': freq.values})
        frequencies.append(freq_df)
    
    freq_df = pd.concat(frequencies, ignore_index=True)
    df = pd.merge(df, freq_df, on=['Source'], how='left')
    df['Frequency'] = df['Frequency'].fillna(0)

    # --- Rate of Change of RSSI ---
    df['RSSI_diff'] = df.groupby('Source')['RSSI'].diff().fillna(0)

    # --- Number of Unique UUIDs Advertised by a Source (in a 5-second window) ---
    def unique_uuids_in_window(group):
        """Calculates the number of unique UUIDs advertised by a source in rolling windows."""
        bins = pd.cut(group['Timestamp'], bins=np.arange(group['Timestamp'].min(), group['Timestamp'].max() + 5, 5))
        return bins.map(lambda x: group.loc[bins == x, 'UUID 16'].nunique())

    uuid_counts = []
    for _, group in df.groupby('Source'):
        counts = unique_uuids_in_window(group)
        uuid_counts.append(pd.Series(counts.values, index=group.index))
    
    df['Unique_UUID_Count_5s'] = pd.concat(uuid_counts).reindex(df.index).fillna(0)

    # --- Entropy of Advertised UUIDs (in a 5-second window)---
    def calculate_entropy(uuids):
        """Calculates the entropy of a list of UUIDs."""
        if len(uuids) <= 1:
            return 0.0
        counts = pd.Series(uuids).value_counts()
        probabilities = counts / len(uuids)
        return -np.sum(probabilities * np.log2(probabilities))

    def entropy_of_uuids(group):
        """Calculates the entropy of UUIDs in rolling windows."""
        bins = pd.cut(group['Timestamp'], bins=np.arange(group['Timestamp'].min(), group['Timestamp'].max() + 5, 5))
        return bins.map(lambda x: calculate_entropy(group.loc[bins == x, 'UUID 16']))

    entropy_values = []
    for _, group in df.groupby('Source'):
        entropy = entropy_of_uuids(group)
        entropy_values.append(pd.Series(entropy.values, index=group.index))
    
    df['UUID_Entropy_5s'] = pd.concat(entropy_values).reindex(df.index).fillna(0)

    # --- Is Company ID Suspicious (Example - Replace with your actual list) ---
    suspicious_company_ids = ['XXXX', 'YYYY', 'ZZZZ']  # Replace with actual suspicious IDs
    df['Company_ID_Suspicious'] = df['Company ID'].isin(suspicious_company_ids).astype(int)

    # --- Save the feature-engineered data ---
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file_name = os.path.basename(cleaned_csv_path).replace("cleansed_", "feature_engineered_")
    output_file_path = os.path.join(output_directory, output_file_name)
    df.to_csv(output_file_path, index=False)
    print(f"Feature engineered data saved to: {output_file_path}")
    return output_file_path

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
