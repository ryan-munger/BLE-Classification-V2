import os
import pandas as pd
import argparse
import os.path
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Constants for labels
LABEL_MALICIOUS = 1  # Malicious label
LABEL_BENIGN = 0  # Benign label

# Model testing and evaluation
def test_model():
    pass

# Model training and evaluation
def train_model(X, y):
    """
    Train a Random Forest model to classify Bluetooth/BLE traffic as Malicious or Benign.

    Args:
        X (DataFrame): Features for training.
        y (Series): Labels for training (0 = Benign, 1 = Malicious).

    Returns:
        RandomForestClassifier: Trained model.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

    return model

# Load and preprocess the data
def load_data(pathToCSV):
    """
    Load the Bluetooth/BLE traffic data from all CSV files in the Downloads folder.

    Returns:
        tuple: Features (X) and labels (y) combined from all files.
    """
    X_list = []
    y_list = []

    print(f"Processing file: {pathToCSV}")

    # Read the file
    data = pd.read_csv(pathToCSV, encoding='latin1')

    # Ensure expected columns are present
    if 'Source' in data.columns and 'Destination' in data.columns and 'Length' in data.columns:
        # Feature selection
        X = data[['Source', 'Destination', 'Length']]  # Modify as needed
        # Assuming the last column is the label
        y = data.iloc[:, -1].apply(lambda val: LABEL_MALICIOUS if val == 'Malicious' else LABEL_BENIGN)

        X_list.append(X)
        y_list.append(y)

    # Combine all data
    X_combined = pd.concat(X_list, axis=0, ignore_index=True)
    y_combined = pd.concat(y_list, axis=0, ignore_index=True)

    return X_combined, y_combined

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

    # Load data
    X, y = load_data(args.csv)

    # Train and evaluate the model
    model = train_model(X, y)

    # Save the model for future use (optional)
    # import joblib
    # joblib.dump(model, "bluetooth_traffic_model.pkl")

# start script
if __name__ == "__main__":
    main()