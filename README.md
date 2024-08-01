# Supervised Learning

Welcome to the Supervised Learning repository. This repository covers the fundamental concepts of supervised learning, including both classification and regression techniques.

## Branches

- [Classification](https://github.com/niladrridas/Supervised-Learning/tree/Classification)
- [Regression](https://github.com/niladrridas/Supervised-Learning/tree/Regression)
- [Setup](https://github.com/niladrridas/Supervised-Learning/tree/gh-pages)

Explore the respective branches for detailed explanations and code examples.

## Topics Covered

1. **Supervised Learning**:
    - Overview and concepts
    - Differences from unsupervised learning

2. **Classification**:
    - Logistic Regression
    - Decision Trees
    - Random Forests
    - Support Vector Machines
    - Evaluation metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC

3. **Regression**:
    - Linear Regression
    - Polynomial Regression
    - Ridge Regression
    - Lasso Regression
    - Evaluation metrics: MSE, MAE, R-squared

4. **Data Preparation**:
    - Generating datasets using NumPy
    - Data cleaning and preprocessing with Pandas
    - Feature selection and engineering

5. **Model Training and Evaluation**:
    - Splitting data into training and testing sets
    - Training models using Scikit-learn
    - Model evaluation and performance metrics

6. **Model Saving and Loading**:
    - Saving models with Joblib
    - Loading and using saved models

## Libraries and Frameworks Used

This repository uses the following Python libraries and frameworks:

- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For implementing machine learning algorithms and model evaluation.
- **Joblib**: For saving trained models.
- **NumPy**: For generating datasets.

### Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/niladrridas/Supervised-Learning.git
    cd Supervised-Learning
    ```

2. **Create a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the Required Libraries**:
    ```bash
    pip install -r requirements.txt
    ```

### Requirements File

`requirements.txt`:
```
pandas
scikit-learn
joblib
numpy
```

## Additional Information

Please explore the respective branches for detailed explanations and code examples related to classification and regression techniques in supervised learning.

### Steps to Update the README.md and Push Changes

1. **Switch to the Main Branch**:
    ```bash
    git checkout main
    ```

2. **Update the README.md**:
    ```bash
    nano README.md
    ```

3. **Add the Provided Content to the README.md File**:
    (Copy and paste the updated README.md content provided above.)

4. **Save the README.md File** and Exit the Editor.

5. **Commit the Changes**:
    ```bash
    git add README.md
    git commit -m "Update README.md with branch links and setup instructions"
    ```

6. **Push the Changes to GitHub**:
    ```bash
    git push origin main
    ```

### Including NumPy in the Classification and Regression Branches

If you haven't done so already, update the scripts to use NumPy for generating datasets. Below are example scripts that incorporate NumPy.

#### Classification Branch Script Update

`scripts/train_logistic_regression.py`:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Generate a synthetic dataset using NumPy
np.random.seed(42)
X = np.random.randn(100, 3)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Convert to DataFrame for consistency
data = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
data['label'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['feature1', 'feature2', 'feature3']], data['label'], test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the trained model
joblib.dump(model, '../models/logistic_regression_model.pkl')
```

#### Regression Branch Script Update

`scripts/train_linear_regression.py`:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Generate a synthetic dataset using NumPy
np.random.seed(42)
X = np.random.randn(100, 3)
y = 3*X[:, 0] + 2*X[:, 1] + X[:, 2] + np.random.randn(100) * 0.5

# Convert to DataFrame for consistency
data = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
data['label'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['feature1', 'feature2', 'feature3']], data['label'], test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Save the trained model
joblib.dump(model, '../models/linear_regression_model.pkl')
```

### Push the Updated Scripts

1. **Switch to the Classification Branch**:
    ```bash
    git checkout classification
    ```

2. **Update the Script**:
    ```bash
    nano scripts/train_logistic_regression.py
    ```

3. **Add the Provided Content to the Script File**:
    (Copy and paste the updated script content provided above.)

4. **Save the Script File** and Exit the Editor.

5. **Commit the Changes**:
    ```bash
    git add scripts/train_logistic_regression.py
    git commit -m "Update logistic regression script to use NumPy for dataset generation"
    ```

6. **Push the Changes to GitHub**:
    ```bash
    git push origin classification
    ```

7. **Switch to the Regression Branch**:
    ```bash
    git checkout regression
    ```

8. **Update the Script**:
    ```bash
    nano scripts/train_linear_regression.py
    ```

9. **Add the Provided Content to the Script File**:
    (Copy and paste the updated script content provided above.)

10. **Save the Script File** and Exit the Editor.

11. **Commit the Changes**:
    ```bash
    git add scripts/train_linear_regression.py
    git commit -m "Update linear regression script to use NumPy for dataset generation"
    ```

12. **Push the Changes to GitHub**:
    ```bash
    git push origin regression
    ```

## More Information

Please explore the respective branches for detailed explanations and code examples related to classification and regression techniques in supervised learning.
