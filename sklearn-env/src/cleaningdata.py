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
        return 0
    return int(res.group(0).replace(':', ''), 16)

# function to clean the data
def cleaning_data(in_csv):
    try:
        # read the csv in to pandas using ISO encoding
        df = pd.read_csv(in_csv, encoding='ISO-8859-1', dtype='string', header=0)
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
        'Entry': 'string',
        'Device Name': 'string',
        'Power Level (dBm)': 'int64',
        'Info': 'string',
        'Label': 'int64'
    }

    # filling columns NaN values
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
    df['Entry'] = df['Entry'].fillna("None")
    df['Device Name'] = df['Device Name'].fillna("Unnamed")
    df['Power Level (dBm)'] = df['Power Level (dBm)'].fillna("-255")
    df['Info'] = df['Info'].fillna("Unknown")
    df['Label'] = df['Label'].fillna("-1")

    # cleaning and converting RSSI and Timestamp specifically
    df['RSSI'] = df['RSSI'].str.replace(' dBm', '').str.strip().astype(int)
    df['Timestamp'] = df['Timestamp'].str.extract(r'(\d+)').astype(float)

    # convert mac address to int
    for mac_column in ['Source', 'Advertising Address']:
        df[mac_column] = df[mac_column].apply(mac_to_int)
        df[mac_column] = df[mac_column].astype('int64')

    # cleaning and converting column types
    for column, dtype in expected_columns.items():
            if column not in ['RSSI', 'Timestamp', 'Source', 'Advertising Address']:
                df[column] = df[column].str.replace(r'[,|-]', '', regex=True)
                df[column] = df[column].astype(dtype)

    # save the cleaned data to the script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_name = os.path.basename(in_csv)
    output_file = os.path.join(script_dir, f"cleansed_{input_file_name}")
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

    # does the path exist?
    if not os.path.exists(args.csv):
        print('Input csv file "{}" does not exist'.format(args.csv),
              file=sys.stderr)
        sys.exit(-1)

    cleaning_data(args.csv) # call cleaning_data to start cleansing

# start script
if __name__ == "__main__":
    main()