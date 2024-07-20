import pandas as pd
import numpy as np

# Define the number of samples and features
n_samples = 1000
n_features = 5

# Define the feature names
feature_names = [f'Feature{i+1}' for i in range(n_features)]

# Generate the feature data
X = np.random.rand(n_samples, n_features)

# Generate the target data (loan amounts)
y = np.random.uniform(1000, 100000, size=n_samples)

# Create a Pandas DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

# Split the data into training and testing sets (80% for training, 20% for testing)
train_size = int(0.8 * n_samples)
train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

# Save the datasets to CSV files
train_df.to_csv('Datasets/loan_train.csv', index=False)
test_df.to_csv('Datasets/loan_test.csv', index=False)