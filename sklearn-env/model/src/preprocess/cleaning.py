import pandas as pd
import os
import re
import numpy as np
from sklearn.impute import SimpleImputer

def validate_mac_address(mac):
    """Validate and convert MAC address to integer."""
    if pd.isna(mac):
        return 0
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', str(mac).lower())
    if res is None:
        return 0
    return int(res.group(0).replace(':', ''), 16)

"""
RSSI values are now properly bounded (-100 to 0 dBm) 
instead of potentially having invalid positive values
"""
def validate_rssi(rssi):
    """Validate RSSI value (typically between -100 and 0 dBm)."""
    if pd.isna(rssi):
        return -127  # Standard value for missing RSSI
    try:
        rssi = int(str(rssi).replace(' dBm', '').strip())
        return max(-100, min(0, rssi))  # Clamp between -100 and 0
    except (ValueError, TypeError):
        return -127

"""
Channel indices are validated (0-39) 
which is crucial for BLE classification
"""
def validate_channel_index(channel):
    """Validate BLE channel index (0-39)."""
    if pd.isna(channel):
        return -1
    try:
        channel = int(channel)
        return channel if 0 <= channel <= 39 else -1
    except (ValueError, TypeError):
        return -1

"""
Power levels are properly bounded (-20 to +10 dBm)
"""
def validate_power_level(power):
    """Validate power level (typically between -20 and +10 dBm)."""
    if pd.isna(power):
        return -255
    try:
        power = int(power)
        return power if -20 <= power <= 10 else -255
    except (ValueError, TypeError):
        return -255

"""
Timestamp is now properly extracted and converted to float
Properly extracts numeric values
Handles various timestamp formats
Returns -1.0 for invalid timestamps
"""
def validate_timestamp(ts):
    """Validate and clean timestamp."""
    if pd.isna(ts):
        return -1.0
    try:
        # Extract numeric part and convert to float
        numeric = re.search(r'(\d+)', str(ts))
        return float(numeric.group(1)) if numeric else -1.0
    except (ValueError, TypeError):
        return -1.0

"""
This function handles missing data in the dataframe.
It drops columns that are completely empty and rows that have missing values in the label column.
It then creates a new column for each column that has missing values and fills them with the mean of the column.
It also converts the column to a string and fills missing values with 'unknown'.
"""
def handle_missing_data(df, label_col='Label'):
    df = df.copy()
    df.dropna(axis=1, how='all', inplace=True)
    if label_col in df.columns:
        df = df[df[label_col].notna()]

    for col in df.columns:
        if col == label_col:
            continue
        df[col + '_missing'] = df[col].isna().astype(int)
        if df[col].dtype in [np.float64, np.int64]:
            imputer = SimpleImputer(strategy='mean')
            df[col] = imputer.fit_transform(df[[col]])
        else:
            df[col] = df[col].astype(str).fillna('unknown')

    return df

"""
This function cleans the raw CSV file.
It reads the CSV file and converts the columns to the correct data types.
It also fills missing values with the appropriate values.
"""
def clean_data(in_csv):
    print("Cleaning raw CSV file...", in_csv)
    try:
        df = pd.read_csv(in_csv, encoding='utf-8-sig', dtype='string', header=0)
    except Exception as e:
        raise RuntimeError(f"Could not read input CSV: {e}")

    expected_columns = {
        'No.': 'int64',
        'Source': 'int64',
        'Destination': 'string',
        'Protocol': 'string',
        'Length': 'int64',
        'Timestamp': 'float64',
        'RSSI': 'int64',
        'Channel Index': 'int64',
        'Advertising Address': 'int64',
        'Company ID': 'string',
        'Packet counter': 'int64',
        'Protocol version': 'int64',
        'UUID 16': 'string',
        'Device Name': 'string',
        'Power Level (dBm)': 'int64',
        'Info': 'string',
        'Label': 'int64'
    }

    # Fill missing values with appropriate defaults
    df['No.'] = df['No.'].fillna("-1")
    df['Source'] = df['Source'].fillna("00:00:00:00:00:00")
    df['Destination'] = df['Destination'].fillna("broadcast")
    df['Protocol'] = df['Protocol'].fillna("Unknown")
    df['Length'] = df['Length'].fillna("-1")
    df['Timestamp'] = df['Timestamp'].fillna("-1.0")
    df['RSSI'] = df['RSSI'].fillna("-127")
    df['Channel Index'] = df['Channel Index'].fillna("-1")
    df['Advertising Address'] = df['Advertising Address'].fillna("00:00:00:00:00:00")
    df['Company ID'] = df['Company ID'].fillna("Unknown")
    df['Packet counter'] = df['Packet counter'].fillna("-1")
    df['Protocol version'] = df['Protocol version'].fillna("-1")
    df['UUID 16'] = df['UUID 16'].fillna("None")
    df['Device Name'] = df['Device Name'].fillna("Unnamed") if 'Device Name' in df.columns else "No Data"
    df['Power Level (dBm)'] = df['Power Level (dBm)'].fillna("-255")
    df['Info'] = df['Info'].fillna("Unknown")
    df['Label'] = df['Label'].fillna("-1")

    # Apply validation functions
    df['RSSI'] = df['RSSI'].apply(validate_rssi)
    df['Timestamp'] = df['Timestamp'].apply(validate_timestamp)
    df['Channel Index'] = df['Channel Index'].apply(validate_channel_index)
    df['Power Level (dBm)'] = df['Power Level (dBm)'].apply(validate_power_level)
    
    for mac_column in ['Source', 'Advertising Address']:
        df[mac_column] = df[mac_column].apply(validate_mac_address)

    # Convert remaining columns to appropriate types
    for column, dtype in expected_columns.items():
        if column not in ['RSSI', 'Timestamp', 'Source', 'Advertising Address', 'Channel Index', 'Power Level (dBm)']:
            df[column] = df[column].str.replace(r'[,|-]', '', regex=True)
            df[column] = df[column].astype(dtype)

    return df