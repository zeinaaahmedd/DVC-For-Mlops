import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess():
    # Load the dataset
    # Note: Using 'MELBOURNE_HOUSE_PRICES_LESS.csv' as per your dataset choice
    raw_path = 'data/MELBOURNE_HOUSE_PRICES_LESS.csv'
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found!")
        return

    df = pd.read_csv(raw_path)

    # Simple cleaning: Drop rows with missing prices as that's our target
    df = df.dropna(subset=['Price'])
    
    # Create a classification target: 1 if Price > Median, 0 otherwise
    median_price = df['Price'].median()
    df['is_expensive'] = (df['Price'] > median_price).astype(int)

    # Select a few numeric features for simplicity
    features = ['Rooms', 'Distance', 'Propertycount']
    # Drop rows with missing feature values
    df = df.dropna(subset=features)
    
    X = df[features]
    y = df['is_expensive']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save processed files
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    print("Preprocessing complete: train.csv and test.csv created.")

if __name__ == "__main__":
    preprocess()