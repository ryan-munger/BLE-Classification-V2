"""
Enhanced Logistic Regression Model Implementation

This module provides an improved Logistic Regression classifier with:
- Hyperparameter tuning using GridSearchCV
- Feature scaling with StandardScaler
- Cross-validation for robust performance estimation
- Comprehensive evaluation metrics
- Model persistence functionality

The implementation is optimized for binary classification tasks and includes
several improvements over the basic scikit-learn implementation.

Key Features:
1. Automated hyperparameter tuning for optimal model performance
2. Automatic feature scaling for better convergence
3. Cross-validation to prevent overfitting
4. Detailed evaluation metrics including ROC-AUC and confusion matrix
5. Easy model saving and loading functionality

Example Usage:
    >>> from ml_model.LogisticRegression import train_logistic_regression
    >>> model, scaler = train_logistic_regression(X_train, y_train)
    >>> metrics = evaluate_logistic_regression(model, scaler, X_test, y_test)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

def train_logistic_regression(X_train, y_train, save_path=None):
    """
    Train an enhanced Logistic Regression model with hyperparameter tuning and cross-validation.
    
    This function implements several improvements over basic Logistic Regression:
    1. Feature scaling using StandardScaler
    2. Grid search for optimal hyperparameters
    3. Cross-validation for robust performance estimation
    4. Model persistence functionality
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix
    y_train : array-like of shape (n_samples,)
        Training target vector
    save_path : str, optional
        Path to save the trained model and scaler. If None, model won't be saved.
    
    Returns:
    --------
    model : LogisticRegression
        Trained Logistic Regression model with optimal hyperparameters
    scaler : StandardScaler
        Fitted StandardScaler instance used for feature scaling
    
    Notes:
    ------
    - The model uses GridSearchCV to find optimal hyperparameters
    - Features are automatically scaled using StandardScaler
    - 5-fold cross-validation is performed during training
    - Both model and scaler are returned to ensure consistent feature scaling
    """
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l1', 'l2'],               # Regularization type
        'solver': ['liblinear', 'saga']        # Optimization algorithm
    }
    
    # Initialize base model
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,              # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1          # Use all available CPU cores
    )
    
    # Train the model
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Print model details
    print("Logistic Regression Model Trained:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    print(f"Model Coefficients: {best_model.coef_}")
    print(f"Intercept: {best_model.intercept_}")
    
    # Save the model and scaler if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump({
            'model': best_model,
            'scaler': scaler
        }, save_path)
        print(f"Model saved to {save_path}")
    
    return best_model, scaler

def evaluate_logistic_regression(model, scaler, X_test, y_test):
    """
    Evaluate the Logistic Regression model using comprehensive metrics.
    
    This function provides detailed evaluation of the model's performance,
    including accuracy, ROC-AUC score, confusion matrix, and classification report.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained Logistic Regression model
    scaler : StandardScaler
        Fitted StandardScaler instance
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix
    y_test : array-like of shape (n_samples,)
        Test target vector
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics:
        - accuracy : float
            Classification accuracy
        - roc_auc : float
            Area under the ROC curve
        - confusion_matrix : array-like
            Confusion matrix
        - classification_report : dict
            Detailed classification metrics
    
    Notes:
    ------
    - Features are scaled using the provided scaler
    - Both predicted classes and probabilities are computed
    - Comprehensive metrics are printed to console
    """
    # Scale the test features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print evaluation metrics
    print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def load_model(model_path):
    """
    Load a saved Logistic Regression model and scaler.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    
    Returns:
    --------
    model : LogisticRegression
        Loaded Logistic Regression model
    scaler : StandardScaler
        Loaded StandardScaler instance
    
    Notes:
    ------
    - The model and scaler must have been saved using train_logistic_regression
    - Both model and scaler are required for consistent feature scaling
    """
    saved_data = joblib.load(model_path)
    return saved_data['model'], saved_data['scaler']
