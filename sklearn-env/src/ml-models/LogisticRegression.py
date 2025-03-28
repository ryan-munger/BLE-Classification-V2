#!/usr/bin/env python3
# -- general imports --
import os
import pandas as pd
import argparse
import sys
import numpy as np

# -- model imports --
from sklearn.linear_model import LogisticRegression

# -- metrics/data imports --
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import RocCurveDisplay

# model calculations and results
def calculate(predictions, y_test):
    cm = confusion_matrix(y_test, predictions)
    TN, FP, FN, TP = cm.ravel()
    print('True Positive(TP)  = ', TP)
    print('False Positive(FP) = ', FP)
    print('True Negative(TN)  = ', TN)
    print('False Negative(FN) = ', FN)
    accuracy =  (TP + TN) / (TP + FP + TN + FN)
    print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))
    print('classification report:')
    print(classification_report(y_test, predictions))
    print('Other accuracy check', accuracy_score(y_test, predictions))

# model training
def train_model(X_train, y_train):
    model = LogisticRegression() # 
    
    model.fit(X_train, y_train)
    return model

# load and preprocess the data
def load_data(csv_file):
    try:
        # read the csv in to pandas using ISO encoding
        dataset = pd.read_csv(csv_file, encoding='ISO-8859-1')
    except:
        print("Was unable to open the file...")
        sys.exit(-1)

    print(f"Processing file: {csv_file}")

    # shuffle the dataset
    shuffledset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)

    X = shuffledset[["Timestamp", "Channel Index", "Advertising Address", "Packet counter", "Power Level (dBm)"]] #select_dtypes(include=[np.number]).iloc[:, :-1] # x is explanatory variables (all columns containing information about the packets except label)
    # X = shuffledset[["Source", "Timestamp", "RSSI", "Channel Index", "Advertising Address", "Packet counter", "Protocol version", "Power Level (dBm)"]]
    y = shuffledset.iloc[:, -1] # y is target variables (in our case it would be the label column)

    return X, y

# saves the trained model
def saveModel(model):
    # Ask the user for a directory to save the model
    directory = input("Enter the directory to save the model: ").strip()

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        create_dir = input("Would you like to create this directory? (yes/no): ").strip().lower()
        if create_dir in ['yes', 'y']:
            os.makedirs(directory)
            print(f"Directory '{directory}' has been created.")
        else:
            print("Model was not saved. Please try again with a valid directory.")
            return

    # Prompt for the filename
    filename = input("Enter the filename to save the model (e.g., 'model.joblib'): ").strip()

    # Construct the full path
    filepath = os.path.join(directory, filename)

    # Save the model
    joblib.dump(model, filepath)
    print(f"Model has been saved successfully at '{filepath}'.")

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
    # args = command_line_args()  # grab command line arguments

    # # does the path exist?
    # if not os.path.exists(args.csv):
    #     print('Input csv file "{}" does not exist'.format(args.csv),
    #           file=sys.stderr)
    #     sys.exit(-1)

    # # load the data
    # X, y = load_data(args.csv)

    # Read csv file
    dataset = pd.read_csv('./test-data/previous-old-data/old_dataset_cleansed.csv')

    # shuffle the dataset
    shuffledset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)

    # Prepare the data & Target column is "Label"
    X = shuffledset[["Timestamp", "Channel Index", "Advertising Address", "Packet counter", "Power Level (dBm)"]] #select_dtypes(include=[np.number]).iloc[:, :-1] 
    y = shuffledset.iloc[:, -1] 

    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=0)

    # normalize the training data -- calculate mean and standard deviation
    ss_train = StandardScaler()
    X_train = ss_train.fit_transform(X_train)
    ss_test = StandardScaler()
    X_test = ss_test.fit_transform(X_test)

    # train
    model = train_model(X_train, y_train)

    # model testing and evaluation
    predictions = model.predict(X_test)

    ax = plt.gca()
    model_graph = RocCurveDisplay.from_estimator(model, X_test, y_test)
    model_graph.plot(ax=ax, alpha=0.8)

    plt.show()

    # calculate metrics and performance
    calculate(predictions, y_test)

    saveModel(model)

# start script
if __name__ == "__main__":
    main()