"""
Script to run Logistic Regression model on the BLE classification dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml_model.LogisticRegression import train_logistic_regression, evaluate_logistic_regression
import os

def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset for training.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    """
    # Load the data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y

def main():
    # Set paths
    data_path = os.path.join('..', 'data', 'old_all_combined.csv')
    model_save_path = os.path.join('..', 'savedModel', 'logistic_regression_model.joblib')
    
    # Load and prepare data
    X, y = load_and_prepare_data(data_path)
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    print("\nTraining Logistic Regression model...")
    model, scaler = train_logistic_regression(
        X_train, y_train,
        save_path=model_save_path
    )
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    metrics = evaluate_logistic_regression(model, scaler, X_test, y_test)
    
    # Print final metrics
    print("\nFinal Model Performance:")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main() 