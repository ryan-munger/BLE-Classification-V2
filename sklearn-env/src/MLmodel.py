#!/usr/bin/env python3
# -- general imports --
import os
import pandas as pd
import argparse
import sys

# -- model imports --
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -- metrics/data imports --
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# label constants
LABEL_MALICIOUS = 1  # malicious
LABEL_BENIGN = 0  # benign

# model testing and evaluation
def test_model():
    pass

# model training
def train_model():
    pass
    

# load and preprocess the data
def load_data(csv_file):
    try:
        # read the csv in to pandas using ISO encoding
        dataset = pd.read_csv(csv_file, encoding='ISO-8859-1')
    except:
        print("Was unable to open the file...")
        sys.exit(-1)
    
    print(dataset.head())

# grab the csv file as a command line argument
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

    # call load data
    load_data(args.csv)

    # train

    # test

# start script
if __name__ == "__main__":
    main()