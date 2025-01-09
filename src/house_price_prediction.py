# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
# Ensure 'house_prices.csv' has columns: SquareFeet, Bedrooms, Age, Price
data = pd.read_csv("../dataset/housing_dataset.csv")

# Display basic information about the dataset
print("Dataset Head:\n", data.head())
print("Dataset Info:\n", data.info())

# Feature and target selection
X = data[["SquareFeet", "Bedrooms", "Age"]]
y = data["Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualize predictions
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")

plt.grid()
plt.show()

# Predict on new data
new_data = pd.DataFrame(
    {"SquareFeet": [2000, 1500], "Bedrooms": [3, 2], "Age": [10, 5]}
)
predictions = model.predict(new_data)
print(f"Predictions for new data:\n{predictions}")
