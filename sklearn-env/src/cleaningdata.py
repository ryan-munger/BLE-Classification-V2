#!/usr/bin/env python3
import pandas as pd
import numpy as np

import argparse
import os.path
import sys


# cleans the data
def cleaning_data(in_csv):
    # read the csv in to pandas using ISO encoding
    df = pd.read_csv(in_csv, encoding='ISO-8859-1')

    # list of columns we want to keep (provided by CS team)
    columns_to_keep = [
        'No.',
        'Timestamp',
        'RSSI',
        'Channel Index',
        'Advertising Address',
        'Company ID',
        'Packet counter',
        'Protocol version',
        'Power Level (dBm)',
        'UUID 16'
    ]

    # select the columns we want
    df = df[columns_to_keep]

    # --- not doing rn ----
    # if column is missing all or over 70% of info (NaN) drop entire column
    #threshold = len(df) * 0.7
    #df = df.dropna(axis=1, how='all')
    #df = df.dropna(axis=1, thresh=threshold)
    # values that are out of range or do not match the expected format should be corrected or replaced with valid value/converted to correct format

    # if less than 70% NaN then fill with mean or mode values respectfully
    df['Timestamp'] = df['Timestamp'].fillna(-1.0)
    df['RSSI'] = df['RSSI'].fillna(1)
    df['Channel Index'] = df['Channel Index'].fillna(-1)
    df['Advertising Address'] = df['Advertising Address'].fillna(-1)
    df['Company ID'] = df['Company ID'].fillna(-1)
    df['Packet counter'] = df['Packet counter'].fillna(-1)
    df['Protocol version'] = df['Protocol version'].fillna(-1)
    df['Power Level (dBm)'] = df['Power Level (dBm)'].fillna(-255)
    df['UUID 16'] = df['UUID 16'].fillna("None")

    print("Data has been cleansed at cleansed_" + str(in_csv))
    df.to_csv("cleansed_" + str(in_csv), index=False)

    # print records
    print(df.head())

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