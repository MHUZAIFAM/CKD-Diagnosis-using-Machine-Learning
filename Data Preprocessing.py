import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
file_path = r"H:\EE SEM VII\Machine Learning\kidney_disease.csv"
kidney_data = pd.read_csv(file_path)

# Drop the 'id' column as it's not relevant, it would simply be counting from 1-400-ishh
kidney_data.drop(columns=['id'], inplace=True)

# 1. Handle Missing Values
# Impute numerical columns with mean
num_cols = kidney_data.select_dtypes(include=['float64', 'int64']).columns
num_imputer = SimpleImputer(strategy='mean')
kidney_data[num_cols] = num_imputer.fit_transform(kidney_data[num_cols])

# Impute categorical columns with mode
cat_cols = kidney_data.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
kidney_data[cat_cols] = cat_imputer.fit_transform(kidney_data[cat_cols])

# 2. Encode Categorical Variables
# Label encode all categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    kidney_data[col] = le.fit_transform(kidney_data[col])
    label_encoders[col] = le  # Store encoder for inverse transformation if needed

# 3. Normalize Numerical Features
scaler = MinMaxScaler()
kidney_data[num_cols] = scaler.fit_transform(kidney_data[num_cols])

# 4. Verify Preprocessing
print("Dataset after preprocessing:")
print(kidney_data.head())
print("\nMissing Values after Preprocessing:")
print(kidney_data.isnull().sum())

# Save the preprocessed dataset to a CSV file
output_path = r"H:\EE SEM VII\Machine Learning\kidney_disease_preprocessed.csv"  # Update with your desired path
kidney_data.to_csv(output_path, index=False)

print(f"Dataset saved to {output_path}")
