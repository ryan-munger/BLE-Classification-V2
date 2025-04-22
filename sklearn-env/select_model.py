import os
import pandas as pd
from src.preprocess.cleaning import clean_data
from src.preprocess.transforming import transform_data
from src.ml_model.LogisticRegression import train_logistic_regression, evaluate_logistic_regression
from src.ml_model.RandomForestClassifier import train_random_forest, evaluate_random_forest
from metrics.MetricChart import evaluate_model  # Import the evaluation function
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and prepare data
def load_data():
    raw_path = './captured/raw_user_data.csv'
    cleaned_path = './data/dataset.csv'

    df = clean_data(raw_path)  # Pass the raw CSV file path here
    df = transform_data(df)
    df.to_csv(cleaned_path, index=False)
    return df

def select_and_train_model(df):

    # Assuming target column is named 'target'
    X = df.drop('Label', axis=1)
    y = df['Label']

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n Class Distribution in Test Set:")
    print(y_test.value_counts())

    # Ask user to select a model
    print("Select a model:")
    print("1. Logistic Regression")
    print("2. Random Forest")
    choice = input("Enter 1 or 2: ")

    model = None
    model_name = ""
    
    if choice == '1':
        # Train Logistic Regression model
        model = train_logistic_regression(X_train, y_train)
        model_name = 'LogisticRegression'
        print(f"Training {model_name}...")
        
    elif choice == '2':
        # Train Random Forest model
        model = train_random_forest(X_train, y_train)
        model_name = 'RandomForestClassifier'
        print(f"Training {model_name}...")
        
    else:
        print("Invalid choice, exiting...")
        return
    
    # Save the trained model
    # Save the trained model
    model_save_path = f'savedModel/{model_name}.joblib'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure the directory exists
    dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")


    # Evaluate the trained model
    print(f"Evaluating {model_name}...")
    evaluate_model(model, X_test, y_test, model_name)

if __name__ == "__main__":
    df = load_data()  # Load and preprocess the data
    select_and_train_model(df)  # Train and evaluate the selected model
