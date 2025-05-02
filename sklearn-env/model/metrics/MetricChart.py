from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import os

def evaluate_model(model, X_test, y_test, model_name):
    # Predictions
    predictions = model.predict(X_test)

    # Compute all metrics
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
    rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    # Print metrics
    print(f"\n Evaluation Metrics for {model_name}:")
    print(f" Accuracy       : {acc:.4f}")
    print(f" Precision      : {prec:.4f}")
    print(f" Recall         : {rec:.4f}")
    print(f" F1 Score       : {f1:.4f}")
    print("\n Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Compute and display the confusion matrix

    # Save the confusion matrix plot
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    os.makedirs("metrics", exist_ok=True)
    plt.savefig(f"metrics/{model_name}_confusion_matrix.png")
    plt.show()

    # metrics_path = f"metrics/{model_name}_metrics.json"
    # with open(metrics_path, "w") as f:
    #     json.dump({
    #         "accuracy": acc,
    #         "precision": precision,
    #         "recall": recall,
    #         "f1_score": f1,
    #         "classification_report": report
    #     }, f, indent=4)
    # print(f"Metrics saved to {metrics_path}")
