import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def validate():
    # Load test data and model
    test_df = pd.read_csv('data/test.csv')
    X_test = test_df.drop('is_expensive', axis=1)
    y_test = test_df['is_expensive']
    
    model = joblib.load('models/model.joblib')

    # Predictions
    preds = model.predict(X_test)

    # Calculate Accuracy
    acc = accuracy_score(y_test, preds)
    with open('metrics.json', 'w') as f:
        json.dump({'accuracy': acc}, f)

    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Affordable', 'Expensive'], 
                yticklabels=['Affordable', 'Expensive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print(f"Validation complete. Accuracy: {acc}")

if __name__ == "__main__":
    validate()