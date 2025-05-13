#!/usr/bin/env python3
# -- general imports --
import os
import pandas as pd
import argparse
import sys
import numpy as np
from pathlib import Path

# -- model imports --
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay, precision_recall_curve, average_precision_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint

# model calculations and results
def calculate(predictions, y_test, probabilities=None):
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    TN, FP, FN, TP = cm.ravel()
    
    # Basic Metrics
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print('\nConfusion Matrix:')
    print(f'True Positive(TP)  = {TP}')
    print(f'False Positive(FP) = {FP}')
    print(f'True Negative(TN)  = {TN}')
    print(f'False Negative(FN) = {FN}')
    
    print('\nPerformance Metrics:')
    print(f'Accuracy  = {accuracy:.4f}')
    print(f'Precision = {precision:.4f}')
    print(f'Recall    = {recall:.4f}')
    print(f'F1 Score  = {f1:.4f}')
    
    print('\nClassification Report:')
    print(classification_report(y_test, predictions))
    
    # Additional metrics if probabilities are provided
    if probabilities is not None:
        avg_precision = average_precision_score(y_test, probabilities)
        print(f'\nAverage Precision Score = {avg_precision:.4f}')

# model training with hyperparameter tuning
def train_model(X_train, y_train):
    # Define parameter grid for hyperparameter tuning
    param_dist = {
        # 'n_estimators': [50, 100, 200],        # Number of trees
        # 'max_depth': [None, 10, 20, 30],       # Maximum depth of trees
        # 'min_samples_split': [2, 5, 10],       # Minimum samples required to split
        # 'min_samples_leaf': [1, 2, 4],         # Minimum samples required at leaf node
        # 'max_features': ['sqrt', 'log2'],      # Number of features to consider
        # 'class_weight': [None, 'balanced'],    # Class weights
        # 'bootstrap': [True, False]             # Bootstrap samples for tree building

        'n_estimators': [100, 200],  # Reduce to 2 values
        'max_depth': [None, 10],      # Limit depth
        'min_samples_split': [2, 5],  # Fewer options
        'min_samples_leaf': [1, 2],   # Fewer options
        'max_features': ['sqrt'],     # Use just one option
        'class_weight': [None],       # Remove 'balanced'
        'bootstrap': [True]           # Only one option
    }
    
    # Initialize base model
    base_model = RandomForestClassifier(random_state=42, n_jobs=2, oob_score=True)
    
    # Perform randomized search for hyperparameter optimization
    print("\nTuning Random Forest hyperparameters...")
    random_search = RandomizedSearchCV(
        # estimator=base_model,
        # param_distributions=param_dist,
        # n_iter=10,               # Number of parameter settings to sample
        # cv=5,                    # 5-fold cross-validation
        # scoring='accuracy',
        # n_jobs=2,               # Use all available CPU cores
        # verbose=1,
        # random_state=42
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=5,               # Reduce to 5 iterations for faster testing
        cv=5,                   # Keep 5-fold cross-validation
        scoring='accuracy',
        n_jobs=2,
        verbose=1,
        random_state=42
    )
    
    # Fit the random search
    random_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    
    # Print tuning results
    print("\nTuning Results:")
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best Cross-Validation Score: {random_search.best_score_:.4f}")
    
    # Perform cross-validation on the best model
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, n_jobs=2)
    print(f"\nCross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return best_model

# load and preprocess the data
def load_data(csv_file):
    try:
        # read the csv in to pandas using ISO encoding
        dataset = pd.read_csv(csv_file, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(-1)

    print(f"Processing file: {csv_file}")

    # shuffle the dataset
    shuffledset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Select features - using all relevant numerical features
    X = shuffledset[["RSSI", "Channel Index", "Company ID", "Protocol version", "Power Level (dBm)"]]
    y = shuffledset.iloc[:, -1]  # Target variable (label column)

    # Print dataset information
    print(f"\nDataset Information:")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")

    return X, y

# saves the trained model
def save_model(model, directory='models'):
    # Create directory if it doesn't exist
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = directory / 'random_forest_model.joblib'
    joblib.dump(model, model_path)
    
    print(f"\nModel saved to {model_path}")

# plot evaluation metrics
def plot_metrics(model, X_test, y_test, predictions, probabilities):
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0, 0])
    axes[0, 0].set_title('ROC Curve')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    axes[0, 1].plot(recall, precision)
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=axes[1, 1])
        axes[1, 1].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.show()

# grab the csv file as a command line argument
def command_line_args():
    parser = argparse.ArgumentParser(description='Train and evaluate a Random Forest model')
    parser.add_argument('--csv', metavar='<input csv file>',
                        help='csv file to parse', required=True)
    parser.add_argument('--output-dir', metavar='<output directory>',
                        help='directory to save the model', default='models')
    args = parser.parse_args()
    return args

# main function
def main():
    args = command_line_args()  # grab command line arguments

    # does the path exist?
    if not os.path.exists(args.csv):
        print('Input csv file "{}" does not exist'.format(args.csv),
              file=sys.stderr)
        sys.exit(-1)

    # load the data
    X, y = load_data(args.csv)

    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # train model with hyperparameter tuning
    model = train_model(X_train, y_train)

    # model testing and evaluation
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # calculate metrics and performance
    calculate(predictions, y_test, probabilities)

    # plot evaluation metrics
    plot_metrics(model, X_test, y_test, predictions, probabilities)

    # save model
    save_model(model, args.output_dir)

# start script
if __name__ == "__main__":
    main()
