import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train():
    # Load training data
    train_df = pd.read_csv('data/train.csv')
    X_train = train_df.drop('is_expensive', axis=1)
    y_train = train_df['is_expensive']

    # CHANGED: Using Random Forest instead of Logistic Regression
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    joblib.dump(model, 'models/model.joblib')
    print("Training complete: models/model.joblib saved (Random Forest).")

if __name__ == "__main__":
    train()