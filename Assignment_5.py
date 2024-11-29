# Load the dataset from the uploaded file
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Read the data
file_path = "C:\\Users\\prashanthp\\Downloads\\bank+marketing\\bank\\bank.csv"
df = pd.read_csv(file_path, delimiter=';')

# Step 2: Select specific columns for analysis
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

# Step 3: Convert categorical variables to dummy numerical values
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)

# Ensure the target variable 'y' is converted into a numerical format
# Though part of Step 5, this is required here as if not done, will cause error in corr()
df3['y'] = df3['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Step 4: Produce a heatmap of correlation coefficients
correlation_matrix = df3.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Heatmap of Correlation Coefficients")
plt.show()

# Step 5: Define the target variable and explanatory variables
y = df3['y']  # Target variable is already numerical
X = df3.drop('y', axis=1)

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Step 8: Confusion matrix and accuracy for Logistic Regression
conf_matrix_log = confusion_matrix(y_test, y_pred_log)
accuracy_log = accuracy_score(y_test, y_pred_log)

# Print results for Logistic Regression
print("Logistic Regression Confusion Matrix:")
print(conf_matrix_log)
print(f"Logistic Regression Accuracy: {accuracy_log:.2f}")

# Step 9: k-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Confusion matrix and accuracy for KNN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Print results for KNN
print("k-Nearest Neighbors Confusion Matrix:")
print(conf_matrix_knn)
print(f"k-Nearest Neighbors Accuracy: {accuracy_knn:.2f}")

# Step 10: Comparison of results
"""
The Logistic Regression model achieved an accuracy of {:.2f}, while the k-Nearest Neighbors 
model achieved an accuracy of {:.2f}. Logistic Regression showed a better performance in 
terms of predictive power, as observed in the confusion matrices and accuracy scores. 
KNN could potentially improve with hyperparameter tuning.
""".format(accuracy_log, accuracy_knn)

