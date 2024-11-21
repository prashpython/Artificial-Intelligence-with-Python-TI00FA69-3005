import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assignment 1 Problem 1
x = np.linspace(-10, 10, 100)  # X values for plotting
y1 = 2 * x + 1
y2 = 2 * x + 2
y3 = 2 * x + 3

plt.figure(figsize=(8, 6))
plt.plot(x, y1, label='y=2x+1', linestyle='-', color='blue')
plt.plot(x, y2, label='y=2x+2', linestyle='--', color='green')
plt.plot(x, y3, label='y=2x+3', linestyle='-.', color='red')
plt.title('Graphs of y=2x+1, y=2x+2, y=2x+3')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

# Assignment 1 Problem 2
x_points = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y_points = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

plt.figure(figsize=(8, 6))
plt.scatter(x_points, y_points, color='black', marker='+')
plt.title('Scatter Plot of Given Points')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True)
plt.show()

# Assignment 1 Problem 3
# Load the data
csv_file_path = "C:\\Users\\prashantp\\Desktop\\python + AI\\weight-height.csv"
data = pd.read_csv(csv_file_path)

# Extracting lengths and weights
lengths_in_inches = data['Height'].values  # Assuming "Height" column represents lengths
weights_in_pounds = data['Weight'].values  # Assuming "Weight" column represents weights

# Convert lengths and weights to metric system
lengths_in_cm = lengths_in_inches * 2.54  # Inches to centimeters
weights_in_kg = weights_in_pounds * 0.453592  # Pounds to kilograms

# Calculate means
mean_length = np.mean(lengths_in_cm)
mean_weight = np.mean(weights_in_kg)

# Draw histogram of lengths in cm
plt.figure(figsize=(8, 6))
plt.hist(lengths_in_cm, bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Lengths (in cm)')
plt.xlabel('Lengths (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Output mean lengths and weights
print("Mean Length: " + str(mean_length))
print("Mean Weight: " + str(mean_weight))

# Assignment 1 Problem 4
A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
A_inv = np.linalg.inv(A)  # Calculate inverse

# Verify A * A_inv and A_inv * A produce identity matrices
identity_1 = np.dot(A, A_inv)
identity_2 = np.dot(A_inv, A)

# Output matrices and identities
print("Matrix:")
print(A)
print("\n")
print("Inverse Matrix:")
print(A_inv)
print("\n")
print("Identity Matrix:")
print(identity_1)
print("\n")
print("Identity Matrix:")
print(identity_2)
