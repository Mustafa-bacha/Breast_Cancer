# train_breast_cancer_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset from CSV
print("Loading breast cancer dataset from CSV...")
df = pd.read_csv('breast_cancer.csv')

# Assuming the CSV has a 'target' column for labels (0 = malignant, 1 = benign)
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target

# Split features and target into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Gradient Boosting model
print("Training Gradient Boosting model...")
gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Evaluate the model
train_accuracy = gb_model.score(X_train_scaled, y_train)
test_accuracy = gb_model.score(X_test_scaled, y_test)
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy: {test_accuracy:.4f}")

# Save the model, scaler, and feature names
print("Saving model and scaler...")
model_package = {
    'model': gb_model,
    'scaler': scaler,
    'feature_names': list(X.columns)
}

with open('breast_cancer_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("Model and scaler saved successfully as 'breast_cancer_model.pkl'")