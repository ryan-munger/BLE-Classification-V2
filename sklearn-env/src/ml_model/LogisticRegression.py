from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    # Initialize the Logistic Regression model with a high max_iter for convergence
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Print model details
    print("Logistic Regression Model Trained:")
    print(f"Model Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    
    return model

def evaluate_logistic_regression(model, X_test, y_test):
    """
    Evaluate the Logistic Regression model using classification metrics.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy
