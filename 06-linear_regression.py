# Assignment 6: Implement Linear Regression Algorithm

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Create the dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [25, 30, 40, 45, 50, 60, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# Step 2: Select independent and dependent variables
X = df[["Study_Hours"]]
y = df["Marks"]


# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Create the Linear Regression model
model = LinearRegression()


# Step 5: Train the model
model.fit(X_train, y_train)


# Step 6: Predict marks
y_pred = model.predict(X_test)

print("\nActual Marks:")
print(y_test.to_list())

print("\nPredicted Marks:")
print(y_pred.round(2))


# Step 7: Display the regression equation
print("\nRegression Equation:")
print(
    f"Marks = {model.intercept_:.2f} "
    f"+ {model.coef_[0]:.2f} * Study_Hours"
)


# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))


# Step 9: Predict marks for a new student
new_student = pd.DataFrame({"Study_Hours": [6]})
predicted_marks = model.predict(new_student)

print(
    "\nPredicted marks for 6 study hours:",
    round(predicted_marks[0], 2)
)


# Step 10: Visualize the regression line
plt.scatter(
    df["Study_Hours"],
    df["Marks"],
    color="blue",
    label="Actual Data"
)

plt.plot(
    df["Study_Hours"],
    model.predict(X),
    color="red",
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression: Study Hours vs Marks")
plt.legend()
plt.grid(True)
plt.show()