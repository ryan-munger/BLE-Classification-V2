import os
import pandas as pd
from src.preprocess.cleaning import clean_data, handle_missing_data
from src.preprocess.transforming import transform_data
from src.preprocess.feature_engineering import feature_engineering
from src.ml_model.LogisticRegression import train_logistic_regression, evaluate_logistic_regression
from src.ml_model.RandomForestClassifier import train_random_forest, evaluate_random_forest
from metrics.MetricChart import evaluate_model
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    """
    Perform all preprocessing steps in sequence:
    1. Clean the data - /src/cleaning.py
    2. Transform the data - /src/transforming.py
    3. Perform feature engineering - 
    
    Returns:
    --------
    df : pandas.DataFrame
        Fully preprocessed dataframe
    """
    # Set paths
    data_path = os.path.join('data', 'old_all_combined.csv')
    cleaned_path = os.path.join('data', 'cleaned_dataset.csv')
    transformed_path = os.path.join('data', 'transformed_dataset.csv')
    feature_engineered_path = os.path.join('data', 'fe_dataset.csv')
    
    # Step 1: Clean the data
    print("\n=== Step 1: Cleaning Data ===")
    print(f"Loading data from {data_path}...")
    df = clean_data(data_path)
    print("Handling missing data...")
    df = handle_missing_data(df)
    print(f"Saving cleaned data to {cleaned_path}...")
    df.to_csv(cleaned_path, index=False)
    
    # Step 2: Transform the data
    print("\n=== Step 2: Transforming Data ===")
    print("Transforming categorical variables...")
    df = transform_data(df)
    print(f"Saving transformed data to {transformed_path}...")
    df.to_csv(transformed_path, index=False)
    
    # Step 3: Feature Engineering
    print("\n=== Step 3: Feature Engineering ===")
    print("Performing feature engineering...")
    feature_engineering(transformed_path, 'data')
    print(f"Loading feature engineered data from {feature_engineered_path}...")
    df = pd.read_csv(feature_engineered_path)
    
    return df

def select_and_train_model(df):
    """
    Train and evaluate the selected model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataframe
    """
    # Prepare features and target
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Encode labels if they are strings
    if y.dtype == 'object':
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print("Label classes:", label_encoder.classes_)

    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nClass Distribution in Test Set:")
    print(pd.Series(y_test).value_counts())

    # Ask user to select a model
    print("\nSelect a model:")
    print("1. Logistic Regression")
    print("2. Random Forest")
    choice = input("Enter 1 or 2: ")

    model = None
    model_name = ""
    
    if choice == '1':
        # Train Logistic Regression model
        print("\nTraining Logistic Regression model...")
        model, scaler = train_logistic_regression(X_train, y_train)
        model_name = 'LogisticRegression'
        
    elif choice == '2':
        # Train Random Forest model
        print("\nTraining Random Forest model...")
        model = train_random_forest(X_train, y_train)
        model_name = 'RandomForestClassifier'
        
    else:
        print("Invalid choice, exiting...")
        return
    
    # Save the trained model
    model_save_path = os.path.join('savedModel', f'{model_name}.joblib')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    dump(model, model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # Evaluate the model
    print(f"\nEvaluating {model_name}...")
    if model_name == 'LogisticRegression':
        metrics = evaluate_logistic_regression(model, scaler, X_test, y_test)
        print("\nFinal Model Performance:")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    else:
        evaluate_model(model, X_test, y_test, model_name)

if __name__ == "__main__":
    # Perform all preprocessing steps
    df = preprocess_data()
    
    # Train and evaluate the selected model
    select_and_train_model(df)
