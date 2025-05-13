#!/usr/bin/env python3
import os
import pandas as pd
import argparse
import sys
import joblib
import datetime

# Clean the data
from src.preprocess.cleaning import clean_data
from src.preprocess.transforming import transform_data

# Load the model and make predictions
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(-1)

# Predict using the loaded model
def predict(model, X):
    print(model.feature_names_in_)
    try:
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        sys.exit(-1)

# Process and predict from CSV
def process_file(input_path, model):
    try:
        print(f"Processing file: {input_path}")
        df = clean_data(input_path)
        if df.empty:
            print(f"Warning: Empty dataframe after cleaning {input_path}")
            return
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['RSSI', 'Channel Index', 'Packet counter', 'Power Level (dBm)']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if df.empty:
            print(f"Warning: Empty dataframe")
            
        df = transform_data(df)

        X = df[["RSSI", "Channel Index", "Company ID", "Protocol version", "Power Level (dBm)"]]
        predictions = model.predict(X)

        # Write predictions to a file
        # Get current date and time
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        # Construct the file path with timestamp
        predictions_file_path = f'./output/predictions_{timestamp}.txt'
        with open(predictions_file_path, 'w') as f:
            for prediction in predictions:
                f.write(str(prediction) + '\n')
        print(f"\nPredictions written to: {predictions_file_path}")
            
    except Exception as e:
        print(f"Processing error: {e}")


# Command line arguments
def command_line_args():
    parser = argparse.ArgumentParser(description='Predict using a saved model')
    parser.add_argument('--csv', required=True, help='Input CSV file')
    parser.add_argument('--model', required=True, help='Path to the saved model')
    return parser.parse_args()

# Main function
def main():
    args = command_line_args()
    model = load_model(args.model)
    process_file(args.csv, model)

if __name__ == "__main__":
    main()
