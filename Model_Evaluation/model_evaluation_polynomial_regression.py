import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the testing dataset
df = pd.read_csv('Datasets/loan_test.csv')

# Split the data into features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Load the trained Polynomial Regression model
model = LinearRegression()
model.load('polynomial_regression_model.pkl')

# Make predictions on the testing set
y_pred = model.predict(X)

# Evaluate the model using MSE, MAE, and R2 score
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R2 score:", r2)