#!/usr/bin/env python3

# --imports--
import pandas as pd
import argparse
import os.path
import sys
import re

# function to convert mac address to int using: https://gist.github.com/nlm/9ec20c78c4881cf23ed132ae59570340
def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
    if res is None:
        return 0  # Changed from "" to 0 for consistency and to avoid potential errors
    return int(res.group(0).replace(':', ''), 16)

# function to clean the data
def cleaning_data(in_csv):
    print("Inside cleanining data function", in_csv)
    try:
        # read the csv in to pandas using ISO encoding
        df = pd.read_csv(in_csv, encoding='ISO-8859-1', header=0)
    except Exception as e:
        print(f"Was unable to open the file... Error: {e}")
        sys.exit(-1)

    # these are the columns we are expecting in the csv file
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

    # filling columns NaN values
    df['No.'] = pd.to_numeric(df['No.'], errors='coerce').fillna(-1).astype('int64')
    df['Source'] = df['Source'].fillna("00:00:00:00:00:00")
    df['Destination'] = df['Destination'].fillna("broadcast")
    df['Protocol'] = df['Protocol'].fillna("Unknown")
    df['Length'] = pd.to_numeric(df['Length'], errors='coerce').fillna(-1).astype('int64')
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce').fillna(-1.0).astype('float64')
    df['RSSI'] = pd.to_numeric(df['RSSI'], errors='coerce').fillna(-255).astype('int64')
    df['Channel Index'] = pd.to_numeric(df['Channel Index'], errors='coerce').fillna(-1).astype('int64')
    df['Advertising Address'] = df['Advertising Address'].fillna("00:00:00:00:00:00")
    df['Company ID'] = df['Company ID'].fillna("Unknown")
    df['Packet counter'] = pd.to_numeric(df['Packet counter'], errors='coerce').fillna(-1).astype('int64')
    df['Protocol version'] = pd.to_numeric(df['Protocol version'], errors='coerce').fillna(-1).astype('int64')
    df['UUID 16'] = df['UUID 16'].fillna("None")
    if 'Device Name' in df.columns:
        df['Device Name'] = df['Device Name'].fillna("Unnamed")
    else:
        df['Device Name'] = "No Data"
    df['Power Level (dBm)'] = pd.to_numeric(df['Power Level (dBm)'], errors='coerce').fillna(-255).astype('int64')
    df['Info'] = df['Info'].fillna("Unknown")
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(-1).astype('int64')

    # cleaning and converting RSSI and Timestamp specifically
    if df['RSSI'].dtype == 'object':
        df['RSSI'] = df['RSSI'].str.replace(' dBm', '').str.strip()
    df['RSSI'] = pd.to_numeric(df['RSSI'], errors='coerce').fillna(-255).astype('int64')
    
    if df['Timestamp'].dtype == 'object':
        df['Timestamp'] = df['Timestamp'].str.extract(r'(\d+\.\d+|\d+)')
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce').fillna(-1.0).astype('float64')

    # convert mac address to int
    for mac_column in ['Source', 'Advertising Address']:
        df[mac_column] = df[mac_column].apply(mac_to_int)
        df[mac_column] = df[mac_column].astype('int64')

    # cleaning and converting column types
    for column, dtype in expected_columns.items():
        if column not in ['RSSI', 'Timestamp', 'Source', 'Advertising Address']:
            if df[column].dtype == 'object':
                df[column] = df[column].str.replace(r'[,|-]', '', regex=True)
            if dtype == 'string':
                df[column] = df[column].astype(str)
            else:
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(-1).astype(dtype)

    # save the cleaned data to the script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_directory = os.path.join(script_dir, './../cleansed_data')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    os.chdir(output_directory)

    input_file_name = os.path.basename(in_csv)
    output_file = os.path.join(output_directory, f"cleansed_{input_file_name}")
    df.to_csv(output_file, index=False)
    print(f"Data has been cleansed and saved at: {output_file}")

    return

# script requires command line argument --csv "file path"
def command_line_args():
    # parses argument input
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', metavar='<input csv file>',
                        help='csv file to parse', required=True)
    args = parser.parse_args()
    return args

# main function
def main():
    args = command_line_args()  # grab command line arguments
    print(args)
    print("Inside main function")

    # does the path exist?
    if not os.path.exists(args.csv):
        print('Input csv file "{}" does not exist'.format(args.csv),
              file=sys.stderr)
        sys.exit(-1)

    cleaning_data(args.csv) # call cleaning_data to start cleansing

# start script
if __name__ == "__main__":
    main()
