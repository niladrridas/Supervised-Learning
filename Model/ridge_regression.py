import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('Datasets/loan_train.csv')

# Split the data into features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Ridge Regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted values:", y_pred)