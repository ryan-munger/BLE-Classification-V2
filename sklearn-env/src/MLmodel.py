import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Constants for labels
LABEL_MALICIOUS = 1  # Malicious label
LABEL_BENIGN = 0  # Benign label

# Directory to load files from
DOWNLOADS_FOLDER = os.path.expanduser("~/Downloads")


# Load and preprocess the data
def load_data():
    """
    Load the Bluetooth/BLE traffic data from all CSV files in the Downloads folder.

    Returns:
        tuple: Features (X) and labels (y) combined from all files.
    """
    X_list = []
    y_list = []

    # Iterate through CSV files in the Downloads folder
    for file in os.listdir(DOWNLOADS_FOLDER):
        if file.endswith(".csv"):
            csv_path = os.path.join(DOWNLOADS_FOLDER, file)
            print(f"Processing file: {csv_path}")

            # Read the file
            data = pd.read_csv(csv_path, encoding='latin1')

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


# Placeholder for MAC address linking
def link_mac_addresses(data):
    """
    Placeholder for functionality to link MAC addresses.
    Add the implementation to correlate MAC addresses here.
    """
    # Example: Group data by MAC addresses to identify relationships
    # mac_groups = data.groupby('Source')
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


# Main script
if __name__ == "__main__":
    # Load data
    X, y = load_data()

    # Link MAC addresses (placeholder functionality)
    link_mac_addresses(X)

    # Train and evaluate the model
    model = train_model(X, y)

    # Save the model for future use (optional)
    # import joblib
    # joblib.dump(model, "bluetooth_traffic_model.pkl")
