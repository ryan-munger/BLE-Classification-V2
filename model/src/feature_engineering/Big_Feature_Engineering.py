import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import itertools
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load the CSV data file."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully! Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(data):
    """Basic data exploration."""
    if 'Label' not in data.columns:
        print("Warning: 'Label' column not found in the dataset.")
        return
    
    print("\n--- Basic Data Information ---")
    print(f"Number of records: {data.shape[0]}")
    print(f"Number of features: {data.shape[1]-1}")  # Excluding Label
    
    print("\n--- Label Distribution ---")
    label_counts = data['Label'].value_counts()
    print(label_counts)

    print("\n--- Data Types and Missing Values ---")
    missing_data = data.isnull().sum()
    data_types = data.dtypes
    missing_percent = (missing_data / len(data)) * 100
    
    info_df = pd.DataFrame({
        'Data Type': data_types,
        'Missing Values': missing_data,
        'Missing Percent': missing_percent.round(2)
    })
    print(info_df)

def analyze_feature_combination_correlation(data, features_to_analyze):
    """Analyze the 'correlation' (using MI and model performance) of combined features with Label."""
    if not all(f in data.columns for f in features_to_analyze + ['Label']):
        print(f"Required columns not found: {', '.join(features_to_analyze)} or Label")
        return
    
    print(f"\n--- 'Correlation' Analysis of {', '.join(features_to_analyze)} vs. Label ---")
    
    X = pd.DataFrame()
    for feature in features_to_analyze:
        if data[feature].dtype in ['int64', 'float64'] and data[feature].nunique() > 10:
            X[feature] = data[feature].fillna(data[feature].median())
        else:
            X = pd.concat([X, pd.get_dummies(data[feature].fillna('missing'), prefix=feature)], axis=1)
            
    y = data['Label']
    
    if X.empty:
        print("No features were processed for analysis.")
        return
        
    # Encode the Label if it's not numerical for mutual information
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Calculate mutual information
    mi_scores = []
    for col in X.columns:
        mi = mutual_info_score(X[col], y_encoded)
        mi_scores.append(mi)
        
    avg_mi = np.mean(mi_scores)
    print(f"\nAverage Mutual Information with Label: {avg_mi:.4f}")
    
    # Train a simple Decision Tree and evaluate with cross-validation
    dt = DecisionTreeClassifier(random_state=42)
    cv_accuracy = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
    print(f"Decision Tree Cross-Validation Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")

def main():
    print("Feature 'Correlation' Analysis Tool - Analyzing combinations of features to determine Label")
    
    # Get file path
    file_path = input("\nEnter the path to your CSV file: ")
    data = load_data(file_path)
    
    if data is None:
        return
        
    # Basic data exploration
    explore_data(data)
    
    # Identify features (exclude Label)
    features = [col for col in data.columns if col != 'Label']
    num_features = len(features)
    
    if 'Label' not in data.columns:
        print("\nWarning: 'Label' column not found in the dataset. Please ensure your target column is named 'Label'.")
        return
        
    while True:
        print("\n--- Analysis Options ---")
        print("1. Analyze a specific combination of up to 5 features")
        print("2. Analyze all combinations of 2 features")
        print("3. Analyze all combinations of 3 features")
        print("4. Analyze all combinations of 4 features")
        print("5. Analyze all combinations of 5 features")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            print("\nAvailable features:")
            for i, feature in enumerate(features, 1):
                print(f"{i}. {feature}")
            
            selected_indices = input("\nEnter the numbers of up to 5 features you want to analyze (comma-separated): ").split(',')
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in selected_indices]
                selected_features = [features[i] for i in selected_indices if 0 <= i < num_features]
                if 1 <= len(selected_features) <= 5:
                    analyze_feature_combination_correlation(data, selected_features)
                else:
                    print("Please select between 1 and 5 valid features.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
                
        elif choice in ['2', '3', '4', '5']:
            n_combinations = int(choice)
            if num_features >= n_combinations:
                print(f"\nAnalyzing all combinations of {n_combinations} features...")
                for combo in itertools.combinations(features, n_combinations):
                    analyze_feature_combination_correlation(data, list(combo))
            else:
                print(f"Not enough features to form combinations of {n_combinations}.")
                
        elif choice == '6':
            print("\nExiting the program. Thank you!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()