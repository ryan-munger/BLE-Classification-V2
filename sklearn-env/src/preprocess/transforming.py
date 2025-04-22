from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def transform_data(df, label_column='Label'):
    df = df.copy()

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found.")

    categorical_fields = [
        'Destination', 'Protocol', 'PHY', 'Company ID',
        'UUID 16', 'Info', 'Device Name', 'Interface description'
    ]

    y = df[label_column].astype(int)
    X = df.drop(columns=[label_column])

    encoders = {}

    for col in X.columns:
        if col in categorical_fields:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    X[label_column] = y
    return X
