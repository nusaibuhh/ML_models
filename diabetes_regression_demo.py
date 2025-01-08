import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

#laod the data
diabetes = datasets.load_diabetes()

#make a dataframe
import pandas as pd
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

df

#feature and target
x = diabetes.data
y = diabetes.target

#split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#initilize the model
model = LinearRegression()

#train the model
model.fit(x_train, y_train)

#predict the model
y_pred = model.predict(x_test)

#evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2:.2f}")

import matplotlib.pyplot as plt

# Assuming you have y_test (actual values) and y_pred (predicted values)

plt.figure(figsize=(8, 6))  # Adjust figure size if needed
plt.scatter(y_test, y_pred, alpha=0.5)  # Scatter plot of actual vs. predicted
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values for Diabetes Regression")
plt.grid(True)
plt.show()