#!/usr/bin/env python3

# --imports--
import pandas as pd
import numpy as np
import argparse
import os.path
import sys

# function to clean the data
def cleaning_data(in_csv):
    try:
        # read the csv in to pandas using ISO encoding
        df = pd.read_csv(in_csv, encoding='ISO-8859-1', low_memory=False)
    except:
        print("Was unable to open the file...")
        sys.exit(-1)

    # these are the columns we are expecting in the csv file
    expected_columns = [
        'No.', 'Source', 'Destination', 'Protocol', 'Length', 'Timestamp', 
        'RSSI', 'Channel Index', 'Advertising Address', 'Company ID', 
        'Packet counter', 'Protocol version', 'UUID 16', 'Entry', 
        'Device Name', 'Power Level (dBm)', 'Info', 'Label'
    ]

    df = df[expected_columns]   # set to make sure

    # fix types
    
    # fill NaN with information
    df['Source'] = df['Source'].fillna("00:00:00:00:00:00")
    df['Destination'] = df['Destination'].fillna("broadcast")
    df['Protocol'] = df['Protocol'].fillna("Unknown")
    df['Length'] = df['Length'].fillna(-1)
    df['Timestamp'] = df['Timestamp'].fillna(-1.0)
    df['RSSI'] = df['RSSI'].fillna(1)
    df['Channel Index'] = df['Channel Index'].fillna(-1)
    df['Advertising Address'] = df['Advertising Address'].fillna("00:00:00:00:00:00")
    df['Company ID'] = df['Company ID'].fillna(-1)
    df['Packet counter'] = df['Packet counter'].fillna(-1)
    df['Protocol version'] = df['Protocol version'].fillna(-1)
    df['UUID 16'] = df['UUID 16'].fillna("None")
    df['Entry'] = df['Entry'].fillna("None")
    df['Device Name'] = df['Device Name'].fillna("Unnamed")
    df['Power Level (dBm)'] = df['Power Level (dBm)'].fillna(-255)
    df['Info'] = df['Info'].fillna("Unknown")
    df['Label'] = df['Label'].fillna(-1)

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