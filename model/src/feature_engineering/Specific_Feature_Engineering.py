import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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
    
    # Plot label distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Label', data=data)
    plt.title('Label Distribution')
    plt.show()
    
    # Display data types and missing values
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

def analyze_single_feature_relation(data, feature):
    """Analyze the relationship between a single feature and the Label."""
    if feature not in data.columns or 'Label' not in data.columns:
        print(f"Required columns not found: {feature} or Label")
        return
    
    print(f"\n--- Analysis of {feature} vs. Label ---")
    
    # Check if feature is numerical or categorical
    if data[feature].dtype in ['int64', 'float64'] and data[feature].nunique() > 10:
        # Numerical feature analysis
        print(f"Feature type: Numerical (unique values: {data[feature].nunique()})")
        
        # Statistical summary by label
        print("\nStatistical summary by label:")
        print(data.groupby('Label')[feature].describe())
        
        # Plot distribution by label
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        for label in data['Label'].unique():
            sns.kdeplot(data[data['Label'] == label][feature], label=f'Label {label}')
        plt.title(f'Distribution of {feature} by Label')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='Label', y=feature, data=data)
        plt.title(f'Boxplot of {feature} by Label')
        
        plt.tight_layout()
        plt.show()
        
        # Calculate mutual information (feature importance)
        mi = mutual_info_score(data[feature].fillna(data[feature].median()), data['Label'])
        print(f"\nMutual Information Score: {mi:.6f}")
        
        # Decision tree feature importance
        X = data[[feature]].fillna(data[feature].median())
        y = data['Label']
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X, y)
        print(f"Decision Tree Feature Importance: {dt.feature_importances_[0]:.6f}")
        
        # Decision tree accuracy
        cv_scores = cross_val_score(dt, X, y, cv=5)
        print(f"Decision Tree CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
    else:
        # Categorical feature analysis
        print(f"Feature type: Categorical (unique values: {data[feature].nunique()})")
        
        # Calculate and display value counts by label
        print("\nValue counts by label:")
        for label in sorted(data['Label'].unique()):
            print(f"\nLabel {label}:")
            print(data[data['Label'] == label][feature].value_counts().head(10))
        
        # Create a contingency table
        contingency = pd.crosstab(data[feature], data['Label'], normalize='index')
        print("\nContingency table (normalized by feature):")
        print(contingency)
        
        # Plot 
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        contingency.plot(kind='bar', stacked=True)
        plt.title(f'Distribution of Labels within each {feature} value')
        plt.xlabel(feature)
        plt.ylabel('Proportion')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.countplot(x=feature, hue='Label', data=data)
        plt.title(f'Count of {feature} by Label')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate mutual information
        mi = mutual_info_score(data[feature].fillna('missing'), data['Label'])
        print(f"\nMutual Information Score: {mi:.6f}")
        
        # Decision tree feature importance
        X = pd.get_dummies(data[feature].fillna('missing'))
        y = data['Label']
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X, y)
        print(f"Decision Tree Average Feature Importance: {dt.feature_importances_.mean():.6f}")
        
        # Decision tree accuracy
        cv_scores = cross_val_score(dt, X, y, cv=5)
        print(f"Decision Tree CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

def analyze_feature_pair_relation(data, feature1, feature2):
    """Analyze how a pair of features relates to the Label."""
    if not all(f in data.columns for f in [feature1, feature2, 'Label']):
        print(f"Required columns not found: {feature1}, {feature2}, or Label")
        return
    
    print(f"\n--- Analysis of {feature1} + {feature2} vs. Label ---")
    
    # Prepare features based on their types
    X = pd.DataFrame()
    
    # Process feature1
    if data[feature1].dtype in ['int64', 'float64'] and data[feature1].nunique() > 10:
        X[feature1] = data[feature1].fillna(data[feature1].median())
    else:
        feat1_dummies = pd.get_dummies(data[feature1].fillna('missing'), prefix=feature1)
        X = pd.concat([X, feat1_dummies], axis=1)
        
    # Process feature2
    if data[feature2].dtype in ['int64', 'float64'] and data[feature2].nunique() > 10:
        X[feature2] = data[feature2].fillna(data[feature2].median())
    else:
        feat2_dummies = pd.get_dummies(data[feature2].fillna('missing'), prefix=feature2)
        X = pd.concat([X, feat2_dummies], axis=1)
    
    y = data['Label']
    
    # Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances for the pair
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importances.head(10))
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
    plt.title(f'Feature Importances for {feature1} + {feature2}')
    plt.tight_layout()
    plt.show()
    
    # Model performance
    cv_scores = cross_val_score(rf, X, y, cv=5)
    print(f"\nRandom Forest CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Try to visualize the relationship if both features are numerical
    if (data[feature1].dtype in ['int64', 'float64'] and data[feature1].nunique() > 10 and
        data[feature2].dtype in ['int64', 'float64'] and data[feature2].nunique() > 10):
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            data[feature1], 
            data[feature2], 
            c=data['Label'], 
            cmap='viridis', 
            alpha=0.6
        )
        plt.colorbar(scatter, label='Label')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'Scatter plot of {feature1} vs {feature2} colored by Label')
        plt.tight_layout()
        plt.show()

def analyze_feature_triple_relation(data, feature1, feature2, feature3):
    """Analyze how three features together relate to the Label."""
    if not all(f in data.columns for f in [feature1, feature2, feature3, 'Label']):
        print(f"Required columns not found: {feature1}, {feature2}, {feature3}, or Label")
        return
    
    print(f"\n--- Analysis of {feature1} + {feature2} + {feature3} vs. Label ---")
    
    # Prepare features based on their types
    X = pd.DataFrame()
    
    # Process feature1
    if data[feature1].dtype in ['int64', 'float64'] and data[feature1].nunique() > 10:
        X[feature1] = data[feature1].fillna(data[feature1].median())
    else:
        feat1_dummies = pd.get_dummies(data[feature1].fillna('missing'), prefix=feature1)
        X = pd.concat([X, feat1_dummies], axis=1)
        
    # Process feature2
    if data[feature2].dtype in ['int64', 'float64'] and data[feature2].nunique() > 10:
        X[feature2] = data[feature2].fillna(data[feature2].median())
    else:
        feat2_dummies = pd.get_dummies(data[feature2].fillna('missing'), prefix=feature2)
        X = pd.concat([X, feat2_dummies], axis=1)
        
    # Process feature3
    if data[feature3].dtype in ['int64', 'float64'] and data[feature3].nunique() > 10:
        X[feature3] = data[feature3].fillna(data[feature3].median())
    else:
        feat3_dummies = pd.get_dummies(data[feature3].fillna('missing'), prefix=feature3)
        X = pd.concat([X, feat3_dummies], axis=1)
    
    y = data['Label']
    
    # Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances for the triple
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importances.head(15))
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
    plt.title(f'Feature Importances for {feature1} + {feature2} + {feature3}')
    plt.tight_layout()
    plt.show()
    
    # Model performance
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print(f"\nRandom Forest CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Try different metrics
    precisions = cross_val_score(rf, X, y, cv=5, scoring='precision_weighted')
    recalls = cross_val_score(rf, X, y, cv=5, scoring='recall_weighted')
    f1s = cross_val_score(rf, X, y, cv=5, scoring='f1_weighted')
    
    print("\nAdditional Metrics:")
    print(f"Precision: {precisions.mean():.4f} ± {precisions.std():.4f}")
    print(f"Recall: {recalls.mean():.4f} ± {recalls.std():.4f}")
    print(f"F1 Score: {f1s.mean():.4f} ± {f1s.std():.4f}")

def main():
    print("Feature Analysis Tool - Focuses on relationships with Label")
    
    # Get file path
    file_path = input("\nEnter the path to your CSV file: ")
    data = load_data(file_path)
    
    if data is None:
        return
        
    # Basic data exploration
    explore_data(data)
    
    # Identify features (exclude Label)
    features = [col for col in data.columns if col != 'Label']
    
    if 'Label' not in data.columns:
        print("\nWarning: 'Label' column not found in the dataset. Please ensure your target column is named 'Label'.")
        return
    
    # Menu loop
    while True:
        print("\n--- Analysis Options ---")
        print("1. Analyze relationship between a single feature and Label")
        print("2. Analyze relationship between a pair of features and Label")
        print("3. Analyze relationship between three features and Label")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            # Single feature analysis
            print("\nAvailable features:")
            for i, feature in enumerate(features, 1):
                print(f"{i}. {feature}")
            
            try:
                feature_idx = int(input("\nSelect a feature (enter the number): ")) - 1
                if 0 <= feature_idx < len(features):
                    analyze_single_feature_relation(data, features[feature_idx])
                else:
                    print("Invalid feature selection.")
            except ValueError:
                print("Please enter a valid number.")
                
        elif choice == '2':
            # Feature pair analysis
            print("\nAvailable features:")
            for i, feature in enumerate(features, 1):
                print(f"{i}. {feature}")
            
            try:
                feat1_idx = int(input("\nSelect first feature (enter the number): ")) - 1
                feat2_idx = int(input("Select second feature (enter the number): ")) - 1
                
                if 0 <= feat1_idx < len(features) and 0 <= feat2_idx < len(features):
                    analyze_feature_pair_relation(data, features[feat1_idx], features[feat2_idx])
                else:
                    print("Invalid feature selection.")
            except ValueError:
                print("Please enter valid numbers.")
                
        elif choice == '3':
            # Feature triple analysis
            print("\nAvailable features:")
            for i, feature in enumerate(features, 1):
                print(f"{i}. {feature}")
            
            try:
                feat1_idx = int(input("\nSelect first feature (enter the number): ")) - 1
                feat2_idx = int(input("Select second feature (enter the number): ")) - 1
                feat3_idx = int(input("Select third feature (enter the number): ")) - 1
                
                if (0 <= feat1_idx < len(features) and 
                    0 <= feat2_idx < len(features) and 
                    0 <= feat3_idx < len(features)):
                    analyze_feature_triple_relation(
                        data, 
                        features[feat1_idx], 
                        features[feat2_idx], 
                        features[feat3_idx]
                    )
                else:
                    print("Invalid feature selection.")
            except ValueError:
                print("Please enter valid numbers.")
                
        elif choice == '4':
            print("\nExiting the program. Thank you!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()