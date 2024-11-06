#!/usr/bin/env python3
import pandas as pd
import numpy as np

import argparse
import os.path
import sys

def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', metavar='<input csv file>',
                        help='csv file to parse', required=True)
    args = parser.parse_args()
    return args

def cleaning_data(in_csv):
    df = pd.read_csv(in_csv)
    df.head()


def main():
    args = command_line_args()

    if not os.path.exists(args.csv):
        print('Input csv file "{}" does not exist'.format(args.csv),
              file=sys.stderr)
        sys.exit(-1)

    cleaning_data(args.csv)

# start script
if __name__ == "__main__":
    main()