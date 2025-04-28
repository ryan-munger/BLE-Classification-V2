import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

def correlation_analysis(df, output_directory):
    """
    Performs correlation analysis on the feature-engineered dataframe.

    Args:
        df (pd.DataFrame): Feature-engineered dataframe.
        output_directory (str): Path to save the correlation heatmap.
    """
    print("\n=== Starting Correlation Analysis ===")

    # Selecting only numeric features for correlation
    numeric_df = df.select_dtypes(include=[np.number])

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    # Plotting heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()

    # Save heatmap
    correlation_plot_path = os.path.join(output_directory, 'feature_correlation_heatmap.png')
    plt.savefig(correlation_plot_path)
    plt.close()

    # Save correlation matrix to CSV
    correlation_csv_path = os.path.join(output_directory, 'feature_correlation_matrix.csv')
    corr_matrix.to_csv(correlation_csv_path)

    print(f"✓ Correlation heatmap saved to: {correlation_plot_path}")
    print(f"✓ Correlation matrix saved to: {correlation_csv_path}")
    print("\n=== Correlation Analysis Complete ===")

    return correlation_csv_path


def main():
    """
    Main function to execute the correlation analysis process.
    """
    parser = argparse.ArgumentParser(description="Perform correlation analysis on feature-engineered BLE data.")
    parser.add_argument('--feature_engineered_csv', required=True, help='Path to the feature-engineered CSV file.')
    parser.add_argument('--output_dir', required=True, help='Path to the directory to save the correlation analysis results.')
    args = parser.parse_args()

    feature_engineered_csv_path = args.feature_engineered_csv
    output_directory = args.output_dir

    if not os.path.exists(feature_engineered_csv_path):
        print(f"Error: Feature-engineered CSV file not found at {feature_engineered_csv_path}")
        sys.exit(1)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the feature-engineered CSV
    try:
        df = pd.read_csv(feature_engineered_csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    # Perform correlation analysis
    correlation_csv_path = correlation_analysis(df, output_directory)
    return correlation_csv_path

if __name__ == "__main__":
    main() 