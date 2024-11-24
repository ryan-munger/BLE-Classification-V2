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
from sklearn.preprocessing import StandardScaler

# label constants
LABEL_MALICIOUS = 1  # malicious
LABEL_BENIGN = 0  # benign

def calculate():
    pass

# model testing and evaluation
def test_model(model, X_test):
    predictions = model.predict(X_test)
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

    print(f"Processing file: {csv_file}")

    X = dataset.iloc[:, :-1]    # x is explanatory variables (all columns containing information about the packets except label)
    y = dataset.iloc[:, -1] # y is target variables (in our case it would be the label column)

    return X, y


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
    X, y = load_data(args.csv)

    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=0)

    # normalize the training data -- calculate mean and standard deviation
    ss_train = StandardScaler()
    X_train = ss_train.fit_transform(X_train)
    ss_test = StandardScaler()
    X_test = ss_test.fit_transform(X_test)

    # train
    model = train_model(X_train, y_train)

    # test
    test_model(model, X_test)

    # calculate metrics and performance
    calculate()

# start script
if __name__ == "__main__":
    main()