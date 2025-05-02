#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from cleaningdata import cleaning_data
from feature_engineering import feature_engineering

def process_captured_data(input_dir, output_dir):
    """
    Process captured BLE data through cleaning, feature engineering, and correlation analysis.
    
    Args:
        input_dir (str): Directory containing captured CSV files
        output_dir (str): Directory to save processed files
    """
    # Create output directories
    cleaned_dir = os.path.join(output_dir, 'cleaned')
    feature_dir = os.path.join(output_dir, 'feature_engineered')
    correlation_dir = os.path.join(output_dir, 'correlation_analysis')
    
    for dir_path in [cleaned_dir, feature_dir, correlation_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process each CSV file in the input directory
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            print(f"\nProcessing file: {file}")
            input_file = os.path.join(input_dir, file)
            
            # Step 1: Clean the data
            print("\n1. Cleaning data...")
            cleaned_file = os.path.join(cleaned_dir, f"cleansed_{file}")
            cleaning_data(input_file)
            
            # Step 2: Feature Engineering
            print("\n2. Performing feature engineering...")
            feature_engineered_file = feature_engineering(cleaned_file, feature_dir)
            
            # Step 3: Correlation Analysis
            print("\n3. Performing correlation analysis...")
            df = pd.read_csv(feature_engineered_file)
            
            # Select numerical columns for correlation
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numerical_cols].corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(15, 12))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title(f'Correlation Matrix - {file}')
            plt.tight_layout()
            
            # Save correlation plot
            correlation_plot = os.path.join(correlation_dir, f"correlation_{file.replace('.csv', '.png')}")
            plt.savefig(correlation_plot)
            plt.close()
            
            # Save correlation matrix to CSV
            correlation_csv = os.path.join(correlation_dir, f"correlation_matrix_{file}")
            correlation_matrix.to_csv(correlation_csv)
            
            print(f"\nProcessing complete for {file}")
            print(f"Cleaned data saved to: {cleaned_file}")
            print(f"Feature engineered data saved to: {feature_engineered_file}")
            print(f"Correlation analysis saved to: {correlation_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process captured BLE data through cleaning, feature engineering, and correlation analysis.')
    parser.add_argument('--input-dir', required=True, help='Directory containing captured CSV files')
    parser.add_argument('--output-dir', required=True, help='Directory to save processed files')
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        sys.exit(1)
    
    process_captured_data(input_dir, output_dir)

if __name__ == "__main__":
    main() 