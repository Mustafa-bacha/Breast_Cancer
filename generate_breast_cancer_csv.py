# generate_breast_cancer_csv.py
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Save to CSV
df.to_csv('breast_cancer.csv', index=False)
print("breast_cancer.csv generated successfully!")