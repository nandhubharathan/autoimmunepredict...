import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset from CSV
df = pd.read_csv('dataset.csv')

# Function to check if a column is binary (has only 2 unique values)
def is_binary(column):
    return len(df[column].unique()) == 2

# Identify categorical columns (assuming they are of object/string type)
categorical_columns = df.select_dtypes(include=['object']).columns

# Separate binary and multi-category columns
binary_columns = [col for col in categorical_columns if is_binary(col)]
multi_category_columns = [col for col in categorical_columns if not is_binary(col)]

# Apply Label Encoding to binary columns
label_encoders = {}
for col in binary_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Apply One-Hot Encoding to multi-category columns
df = pd.get_dummies(df, columns=multi_category_columns)

# Show the transformed data
print(df.head())

# Optional: Save the preprocessed data to a new CSV file if needed
df.to_csv('conv.csv', index=False)
