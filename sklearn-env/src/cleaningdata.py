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

    # print records
    print(df.head())

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