import os
import joblib


def save_model(model):
    # Ask the user for a directory to save the model
    directory = input("Enter the directory to save the model: ").strip()

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        create_dir = input("Would you like to create this directory? (yes/no): ").strip().lower()
        if create_dir in ['yes', 'y']:
            os.makedirs(directory)
            print(f"Directory '{directory}' has been created.")
        else:
            print("Model was not saved. Please try again with a valid directory.")
            return

    # Prompt for the filename
    filename = input("Enter the filename to save the model (e.g., 'model.joblib'): ").strip()

    # Construct the full path
    filepath = os.path.join(directory, filename)

    # Save the model
    joblib.dump(model, filepath)
    print(f"Model has been saved successfully at '{filepath}'.")



    # Call the function
    save_model(model)
