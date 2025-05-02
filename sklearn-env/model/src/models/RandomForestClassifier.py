#!/usr/bin/env python3

# -- general imports --
import os
import pandas as pd
import argparse
import sys
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import logging
from datetime import datetime
import psutil

# -- model imports --
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
def setup_logging(log_file="training_log.txt"):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging initialized")

# Memory tracking
def log_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # in MB
    logging.info(f"{note} - Memory usage: {mem:.2f} MB")

def calculate(predictions, y_test, probabilities=None):
    logging.info("Calculating evaluation metrics...")
    cm = confusion_matrix(y_test, predictions)
    TN, FP, FN, TP = cm.ravel()

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logging.info(f"Confusion Matrix: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
    logging.info(f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")

    if probabilities is not None:
        roc_auc = roc_auc_score(y_test, probabilities)
        avg_precision = average_precision_score(y_test, probabilities)
        logging.info(f"ROC AUC = {roc_auc:.4f}, Average Precision = {avg_precision:.4f}")

    report = classification_report(y_test, predictions)
    logging.info(f"Classification Report:\n{report}")

def train_model(X_train, y_train):
    logging.info("Starting model training...")
    log_memory_usage("Before training")

    param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(10, 30),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced'],
        'bootstrap': [True]
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=2)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    try:
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=20,
            scoring='roc_auc',
            cv=cv,
            verbose=2,
            random_state=42,
            n_jobs=1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        logging.info(f"Best Parameters: {search.best_params_}")
        logging.info(f"Best Cross-Validation Score: {search.best_score_:.4f}")

        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc')
        logging.info(f"Cross-Validation Scores: {cv_scores}")
        logging.info(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        logging.info(f"Top 10 Features:\n{feature_importance.head(10)}")

        log_memory_usage("After training")
        return best_model

    except MemoryError:
        logging.error("MemoryError occurred during model training.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)

    # --- ORIGINAL GRID SEARCH (COMMENTED OUT) ---
    """
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=2,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    """

def load_data(csv_file):
    try:
        logging.info(f"Loading data from: {csv_file}")
        dataset = pd.read_csv(csv_file, encoding='ISO-8859-1')
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        sys.exit(-1)

    shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    X = shuffled[["Source", "Timestamp", "RSSI", "Channel Index", 
                  "Advertising Address", "Packet counter", 
                  "Protocol version", "Power Level (dBm)"]]
    y = shuffled.iloc[:, -1]

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        logging.info(f"Encoded column: {col}")

    logging.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    logging.info(f"Target distribution:\n{y.value_counts(normalize=True)}")

    return X, y

def save_model(model, directory='models'):
    Path(directory).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(directory) / f'random_forest_model_{timestamp}.joblib'
    joblib.dump(model, path)
    logging.info(f"Model saved to {path}")

def plot_metrics(model, X_test, y_test, predictions, probabilities):
    logging.info("Plotting metrics...")
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
        importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=importance.head(10), ax=axes[1, 1])
        axes[1, 1].set_title('Top 10 Feature Importance')

    plt.tight_layout()
    plt.show()

def command_line_args():
    parser = argparse.ArgumentParser(description='Train Random Forest with memory logging')
    parser.add_argument('--csv', required=True, help='CSV input file')
    parser.add_argument('--output-dir', default='models', help='Directory to save model')
    return parser.parse_args()

def main():
    setup_logging()
    args = command_line_args()

    if not os.path.exists(args.csv):
        logging.error(f"Input file '{args.csv}' does not exist")
        sys.exit(-1)

    logging.info("Pipeline started")
    X, y = load_data(args.csv)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    calculate(predictions, y_test, probabilities)
    plot_metrics(model, X_test, y_test, predictions, probabilities)
    save_model(model, args.output_dir)

if __name__ == "__main__":
    main()
