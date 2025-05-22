import pandas as pd
import os

def load_data(path='1_data/raw/churn.csv'):
    return pd.read_csv(path)

def clean_data(df):
    # Convert TotalCharges to numeric (some are empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows where TotalCharges couldn't be converted
    df = df.dropna(subset=['TotalCharges'])

    # Drop customerID
    df = df.drop(columns=['customerID'])

    # Convert Churn to binary: Yes -> 1, No -> 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def engineer_features(df):
    # Total Spending = Monthly Charges * Tenure
    df['TotalSpending'] = df['MonthlyCharges'] * df['tenure']

    # Tenure groups: new, loyal, veteran
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 72],
        labels=['new', 'loyal', 'veteran']
    )

    return df

def encode_categoricals(df):
    # Select all object columns (i.e., categorical)
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # One-hot encode all categorical variables without dropping any
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    return df

def save_data(df, path='1_data/processed/cleaned_telco.csv'):
    # Make sure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save to CSV
    df.to_csv(path, index=False)

if __name__ == '__main__':
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    save_data(df)
    print("Data cleaned and saved to '1_data/processed/cleaned_telco.csv'")
