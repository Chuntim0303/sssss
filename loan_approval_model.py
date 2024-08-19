import os
os.getcwd()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

def load_data(filepath):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Adjust numerical values
    df['income_annum'] = df['income_annum'] / 10
    df['loan_amount'] = df['loan_amount'] / 10

    # Drop unused columns
    df.drop(columns=['loan_id'], inplace=True)

    # Strip whitespace and map categorical values
    df['education'] = df['education'].str.strip()
    df['self_employed'] = df['self_employed'].str.strip()
    df['loan_status'] = df['loan_status'].str.strip()

    # Manual mapping for encoding
    mappings = {
        'education': {'Not Graduate': 0, 'Graduate': 1},
        'self_employed': {'No': 0, 'Yes': 1},
        'loan_status': {'Rejected': 0, 'Approved': 1}
    }

    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    return df

def preprocess_and_train(df):
    """Preprocess the data and train the RandomForest model."""
    # Define features and target variable
    X = df.drop('loan_status', axis=1)  # Features
    y = df['loan_status']  # Target variable

    # Perform train-test split
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Drop missing values
    X_train = X_train.dropna()
    y_train = y_train[X_train.index]  # Ensure y_train is also filtered
    X_test = X_test.dropna()
    y_test = y_test[X_test.index]  # Ensure y_test is also filtered

    # Initialize RandomForestClassifier
    model = RandomForestClassifier(random_state=random_state)

    # Define preprocessing and model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', 'passthrough'),  # No additional preprocessing in this case
        ('classifier', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    return pipeline

def save_model(model, filename='random_forest_model_loan_application.pkl'):
    """Save the trained model to a file."""
    joblib.dump(model, filename)

def load_model(filename='random_forest_model_loan_application.pkl'):
    """Load the trained model from a file."""
    return joblib.load(filename)

def predict(model, user_input):
    """Make predictions using the trained model."""
    # Convert user input to a DataFrame with same structure as training data
    input_df = pd.DataFrame([user_input], columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'])
    
    # Predict using the model
    return model.predict(input_df)

if __name__ == "__main__":
    filepath = r"C:\Users\chunt\OneDrive\Desktop\New folder\loan_approval_dataset.csv"  # Adjust this path as needed
    df = load_data(filepath)
    model = preprocess_and_train(df)
    save_model(model)
    print("Model trained and saved successfully.")

