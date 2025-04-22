import pandas as pd
import os
import re
import numpy as np
from sklearn.impute import SimpleImputer

def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', str(mac).lower())
    if res is None:
        return 0
    return int(res.group(0).replace(':', ''), 16)

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

    df['No.'] = df['No.'].fillna("-1")
    df['Source'] = df['Source'].fillna("00:00:00:00:00:00")
    df['Destination'] = df['Destination'].fillna("broadcast")
    df['Protocol'] = df['Protocol'].fillna("Unknown")
    df['Length'] = df['Length'].fillna("-1")
    df['Timestamp'] = df['Timestamp'].fillna("-1.0")
    df['RSSI'] = df['RSSI'].fillna("1")
    df['Channel Index'] = df['Channel Index'].fillna("-1")
    df['Advertising Address'] = df['Advertising Address'].fillna("00:00:00:00:00:00")
    df['Company ID'] = df['Company ID'].fillna("Unknown")
    df['Packet counter'] = df['Packet counter'].fillna("-1")
    df['Protocol version'] = df['Protocol version'].fillna("-1")
    df['UUID 16'] = df['UUID 16'].fillna("None")

    if 'Device Name' in df.columns:
        df['Device Name'] = df['Device Name'].fillna("Unnamed")
    else:
        df['Device Name'] = "No Data"

    df['Power Level (dBm)'] = df['Power Level (dBm)'].fillna("-255")
    df['Info'] = df['Info'].fillna("Unknown")
    df['Label'] = df['Label'].fillna("-1")

    df['RSSI'] = df['RSSI'].str.replace(' dBm', '', regex=False).str.strip().astype(int)
    df['Timestamp'] = df['Timestamp'].str.extract(r'(\d+)').astype(float)

    for mac_column in ['Source', 'Advertising Address']:
        df[mac_column] = df[mac_column].apply(mac_to_int).astype('int64')

    for column, dtype in expected_columns.items():
        if column not in ['RSSI', 'Timestamp', 'Source', 'Advertising Address']:
            df[column] = df[column].str.replace(r'[,|-]', '', regex=True)
            df[column] = df[column].astype(dtype)

    df = handle_missing_data(df, label_col='Label')

    return df