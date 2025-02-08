# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the California housing dataset
housing = fetch_california_housing()

# Convert the dataset to a DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add the target column (house prices)
df['Price'] = housing.target

# We will use one feature for simplicity, let's use 'AveRooms' (Average number of rooms per household)
X = df[['AveRooms']]  # Independent variable (feature)
y = df['Price']  # Dependent variable (target)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plotting the results
plt.scatter(X_test, y_test, color='blue', label='Actual prices')
plt.plot(X_test, y_pred, color='red', label='Predicted prices')
plt.xlabel('Average number of rooms')
plt.ylabel('Price of houses')
plt.legend()
plt.show()
