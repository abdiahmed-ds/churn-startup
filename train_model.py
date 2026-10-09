import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Automatically find the directory where train_model.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Generating customer churn dataset...")
np.random.seed(42)
n_samples = 1000

# Create realistic customer subscription features
data = {
    'tenure': np.random.randint(1, 72, size=n_samples),
    'MonthlyCharges': np.random.uniform(20.0, 120.0, size=n_samples),
    'TotalCharges': np.random.uniform(20.0, 8000.0, size=n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer'], size=n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples)
}

df = pd.DataFrame(data)

# Simulate churn target based on features
churn_prob = (df['MonthlyCharges'] / 150) + (df['Contract'] == 'Month-to-month').astype(int) * 0.4
df['Churn'] = (churn_prob > np.random.uniform(0.5, 1.2, size=n_samples)).astype(int)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first=True)
model_columns = list(X.columns)

# Train the model
print("Training Random Forest Classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model artifacts securely
model_path = os.path.join(BASE_DIR, "churn_model.pkl")
columns_path = os.path.join(BASE_DIR, "model_columns.pkl")

joblib.dump(model, model_path)
joblib.dump(model_columns, columns_path)

print(f"Success! Model saved to: {model_path}")
print(f"Columns saved to: {columns_path}")