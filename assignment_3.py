import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Assignment 3 Problem 1
n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    dice_throws = np.random.randint(1, 7, size=(n, 2))  # Simulating n throws of two dice
    sums = dice_throws.sum(axis=1)  # Sum of the two dice
    h, h2 = np.histogram(sums, bins=range(2, 14))  # Calculate histogram
    plt.bar(h2[:-1], h / n)  # Plot normalized histogram
    plt.title(f"Histogram of Dice Sums for n = {n}")
    plt.xlabel("Sum of Dice")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# Assignment 3 Problem 2
data = pd.read_csv("C:\\Users\\nachi\\Desktop\\Prashant Palle\\python + AI\\weight-height.csv")  # Replace with the correct path if necessary

X = data[['Height']].values  # Independent variable
y = data['Weight'].values  # Dependent variable

plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.5)
plt.title("Height vs Weight")
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.grid(True)
plt.show()

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.5, label="Actual Data")
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.title("Linear Regression: Height vs Weight")
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.legend()
plt.grid(True)
plt.show()

rmse = mean_squared_error(y, y_pred, squared=False)  # Root Mean Squared Error
r2 = r2_score(y, y_pred)  # R-squared value

print(f"RMSE: {rmse}")
print(f"R²: {r2}")
