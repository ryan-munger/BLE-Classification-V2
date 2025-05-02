#!/usr/bin/env python3
# -- general imports --
import os
import pandas as pd
import argparse
import sys
import numpy as np
from pathlib import Path
import warnings
import time
import psutil
from datetime import datetime
warnings.filterwarnings('ignore')

# -- model imports --
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Memory usage logger
def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {note}Memory usage: {mem:.2f} MB")

# model calculations and results
def calculate(predictions, y_test, probabilities=None):
    cm = confusion_matrix(y_test, predictions)
    TN, FP, FN, TP = cm.ravel()

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

    if probabilities is not None:
        roc_auc = roc_auc_score(y_test, probabilities)
        print(f'ROC AUC   = {roc_auc:.4f}')

    print('\nClassification Report:')
    print(classification_report(y_test, predictions))

    if probabilities is not None:
        avg_precision = average_precision_score(y_test, probabilities)
        print(f'\nAverage Precision Score = {avg_precision:.4f}')

# model training with hyperparameter tuning
def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced'],
        'bootstrap': [True],
        'oob_score': [True]
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=2)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=2,
        verbose=2  # Shows detailed progress
    )

    print("\nTuning Random Forest hyperparameters...")
    print_memory_usage("Before GridSearchCV: ")

    try:
        start = time.time()
        grid_search.fit(X_train, y_train)
        end = time.time()
    except MemoryError:
        print("MemoryError: Not enough RAM during GridSearchCV.")
        sys.exit(-1)

    print_memory_usage("After GridSearchCV: ")
    print(f"\nHyperparameter tuning completed in {end - start:.2f} seconds.")

    best_model = grid_search.best_estimator_

    print("\nTuning Results:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"\nCross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    return best_model

# load and preprocess the data
def load_data(csv_file):
    try:
        dataset = pd.read_csv(csv_file, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(-1)

    print(f"\nProcessing file: {csv_file}")
    print_memory_usage("After reading CSV: ")

    shuffledset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    X = shuffledset[["Source", "Timestamp", "RSSI", "Channel Index", 
                    "Advertising Address", "Packet counter", 
                    "Protocol version", "Power Level (dBm)"]]
    y = shuffledset.iloc[:, -1]

    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"\nEncoding categorical features: {list(categorical_cols)}")
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    print(f"\nDataset Information:")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")

    return X, y

# saves the trained model
def save_model(model, directory='models'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = directory / f'random_forest_model_{timestamp}.joblib'
    joblib.dump(model, model_path)

    print(f"\nModel saved to {model_path}")

# plot evaluation metrics
def plot_metrics(model, X_test, y_test, predictions, probabilities):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0, 0])
    axes[0, 0].set_title('ROC Curve')

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    axes[0, 1].plot(recall, precision)
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')

    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')

    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=axes[1, 1])
        axes[1, 1].set_title('Top 10 Feature Importance')

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
    args = command_line_args()

    if not os.path.exists(args.csv):
        print(f'Input csv file "{args.csv}" does not exist', file=sys.stderr)
        sys.exit(-1)

    print_memory_usage("Startup: ")

    X, y = load_data(args.csv)

    print_memory_usage("Before train-test split: ")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print_memory_usage("After train-test split: ")

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    calculate(predictions, y_test, probabilities)

    plot_metrics(model, X_test, y_test, predictions, probabilities)

    save_model(model, args.output_dir)

    print_memory_usage("End of script: ")

# start script
if __name__ == "__main__":
    main()
