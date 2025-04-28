from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.
    """
    # Initialize the Random Forest Classifier model with default parameters
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    
    # Print model details
    print("Random Forest Classifier Model Trained:")
    print(f"Number of Estimators: {model.n_estimators}")
    print(f"Max Depth: {model.max_depth}")
    
    return model

def evaluate_random_forest(model, X_test, y_test):
    """
    Evaluate the Random Forest model using classification metrics.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    importances = model.feature_importances_
    features = X_train.columns

    indices = importances.argsort()[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig("metrics/feature_importances.png")
    plt.show()

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))



    return accuracy
