import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train():
    # Load training data
    train_df = pd.read_csv('data/train.csv')
    X_train = train_df.drop('is_expensive', axis=1)
    y_train = train_df['is_expensive']

    # We start with Logistic Regression for the 'main' branch
    # Note: In the next steps, we will change this for different branches
    model = LogisticRegression(random_state=42)
    
    model.fit(X_train, y_train)

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    joblib.dump(model, 'models/model.joblib')
    print("Training complete: models/model.joblib saved.")

if __name__ == "__main__":
    train()