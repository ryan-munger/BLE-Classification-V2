# model training
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

# load and preprocess the data
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