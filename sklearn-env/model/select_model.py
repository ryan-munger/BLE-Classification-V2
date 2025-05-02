import os
import pandas as pd
from src.preprocess.cleaning import clean_data
from src.preprocess.transforming import transform_data
from src.preprocess.feature_engineering import feature_engineering
from src.models.LogisticRegression import train_model as train_lr, calculate as calculate_lr
from src.models.RandomForestClassifier import train_model as train_rf, calculate as calculate_rf
from metrics.MetricChart import evaluate_model
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data():
    """
    Perform all preprocessing steps in sequence:
    1. Clean the data from captured folder
    2. Transform the data from cleansed folder
    3. Perform feature engineering on transformed data
    
    Returns:
    --------
    df : pandas.DataFrame
        Fully preprocessed dataframe
    """
    # Set paths
    captured_path = os.path.join('data', 'captured')
    cleansed_path = os.path.join('data', 'cleansed')
    transformed_path = os.path.join('data', 'transformed')
    feature_engineered_path = os.path.join('data', 'fe_dataset.csv')
    
    # Create necessary directories if they don't exist
    os.makedirs(cleansed_path, exist_ok=True)
    os.makedirs(transformed_path, exist_ok=True)
    
    # Verify captured directory exists and has files
    if not os.path.exists(captured_path):
        raise FileNotFoundError(f"Captured data directory not found: {captured_path}")
    
    captured_files = [f for f in os.listdir(captured_path) if f.endswith('.csv')]
    if not captured_files:
        raise FileNotFoundError(f"No CSV files found in {captured_path}")
    
    # Step 1: Clean the data
    print("\n=== Step 1: Cleaning Data ===")
    print(f"Processing {len(captured_files)} files from {captured_path}...")
    
    # Process all CSV files in the captured folder
    for i, filename in enumerate(captured_files, 1):
        try:
            print(f"\nProcessing file {i}/{len(captured_files)}: {filename}")
            input_path = os.path.join(captured_path, filename)
            output_path = os.path.join(cleansed_path, f'cleaned_{filename}')
            
            # Clean the data
            df = clean_data(input_path)
            if df.empty:
                print(f"Warning: Empty dataframe after cleaning {filename}")
                continue
                
            
            # Save cleaned data
            print(f"Saving cleaned data to {output_path}...")
            df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Step 2: Transform the data
    print("\n=== Step 2: Transforming Data ===")
    cleaned_files = [f for f in os.listdir(cleansed_path) if f.startswith('cleaned_')]
    print(f"Transforming {len(cleaned_files)} cleaned files...")
    
    # Process all cleaned files
    for i, filename in enumerate(cleaned_files, 1):
        try:
            print(f"\nTransforming file {i}/{len(cleaned_files)}: {filename}")
            input_path = os.path.join(cleansed_path, filename)
            output_path = os.path.join(transformed_path, f'transformed_{filename.replace("cleaned_", "")}')
            
            # Transform the data with explicit handling of mixed types
            print("Reading CSV file with mixed type handling...")
            df = pd.read_csv(input_path, low_memory=False)
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['RSSI', 'Channel Index', 'Packet counter', 'Power Level (dBm)']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if df.empty:
                print(f"Warning: Empty dataframe when transforming {filename}")
                continue
                
            df = transform_data(df)
            
            # Save transformed data
            print(f"Saving transformed data to {output_path}...")
            df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"Error transforming {filename}: {str(e)}")
            continue
    
    
    # Step 3: Feature Engineering
    # print("\n=== Step 3: Feature Engineering ===")
    # transformed_files = [f for f in os.listdir(transformed_path) if f.startswith('transformed_')]
    # print(f"Performing feature engineering on {len(transformed_files)} transformed files...")
    
    # # Process all transformed files
    # for i, filename in enumerate(transformed_files, 1):
    #     try:
    #         print(f"\nFeature engineering file {i}/{len(transformed_files)}: {filename}")
    #         input_path = os.path.join(transformed_path, filename)
    #         feature_engineering(input_path, 'data')
    #     except Exception as e:
    #         print(f"Error in feature engineering for {filename}: {str(e)}")
    #         continue
    
    # # Load and validate final feature engineered dataset
    # if not os.path.exists(feature_engineered_path):
    #     raise FileNotFoundError(f"Feature engineered dataset not found at {feature_engineered_path}")
        
    # print(f"Loading feature engineered data from {feature_engineered_path}...")
    # df = pd.read_csv(feature_engineered_path, low_memory=False)
    
    # if df.empty:
    #     raise ValueError("Feature engineered dataset is empty")
        
    # if 'Label' not in df.columns:
    #     raise ValueError("Feature engineered dataset missing 'Label' column")
    
    return df

def select_and_train_model(df):
    """
    Train and evaluate the selected model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataframe
    """
    # Prepare features and target
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Encode labels if they are strings
    if y.dtype == 'object':
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print("Label classes:", label_encoder.classes_)

    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nClass Distribution in Test Set:")
    print(pd.Series(y_test).value_counts())

    # Ask user to select a model
    print("\nSelect a model:")
    print("1. Logistic Regression")
    print("2. Random Forest")
    choice = input("Enter 1 or 2: ")

    model = None
    model_name = ""
    
    if choice == '1':
        # Train Logistic Regression model
        print("\nTraining Logistic Regression model...")
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = train_lr(X_train_scaled, y_train)
        model_name = 'LogisticRegression'
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        calculate_lr(predictions, y_test, probabilities)
        
    elif choice == '2':
        # Train Random Forest model
        print("\nTraining Random Forest model...")
        model = train_rf(X_train, y_train)
        model_name = 'RandomForestClassifier'
        
        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        calculate_rf(predictions, y_test, probabilities)
        
    else:
        print("Invalid choice, exiting...")
        return
    
    # Save the trained model
    model_save_path = os.path.join('savedModel', f'{model_name}.joblib')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    dump(model, model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    # Perform all preprocessing steps
    df = preprocess_data()
    
    # Train and evaluate the selected model
    select_and_train_model(df)
