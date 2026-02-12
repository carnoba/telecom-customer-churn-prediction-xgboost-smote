import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """
    Load the Telco-Customer-Churn dataset and perform initial preprocessing.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Handle missing values in TotalCharges
    # TotalCharges is currently an object because of blank spaces
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Requirement: Median Imputation for TotalCharges
    median_val = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
    print(f"Imputed {df['TotalCharges'].isna().sum()} missing values in TotalCharges with median: {median_val}")
    
    # Drop customerID as it's not a feature
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # Requirement: Label Encoding for categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        print(f"Label Encoded: {col}")
    
    # Define features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Requirement: 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns
