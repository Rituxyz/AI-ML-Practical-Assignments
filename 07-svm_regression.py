# Assignment 7: Implement SVM Algorithm for Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Create the dataset
data = {
    "Area": [
        500, 600, 700, 800, 900,
        1000, 1100, 1200, 1300, 1400,
        1500, 1600, 1700, 1800, 1900
    ],
    "Price": [
        15, 18, 20, 23, 25,
        28, 30, 33, 35, 38,
        40, 43, 45, 48, 50
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# Step 2: Select independent and dependent variables
X = df[["Area"]]
y = df["Price"]


# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 4: Create the SVR model
# StandardScaler scales the input features before SVR.
model = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=100, epsilon=0.1)
)


# Step 5: Train the model
model.fit(X_train, y_train)


# Step 6: Predict prices
y_pred = model.predict(X_test)

print("\nActual Prices:")
print(y_test.to_list())

print("\nPredicted Prices:")
print(y_pred.round(2))


# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))


# Step 8: Predict the price for a new house
new_house = pd.DataFrame({"Area": [1250]})

predicted_price = model.predict(new_house)

print(
    "\nPredicted price for a house of area 1250:",
    round(predicted_price[0], 2),
    "lakh"
)


# Step 9: Visualize actual data and SVR predictions
area_range = pd.DataFrame({
    "Area": range(
        int(df["Area"].min()),
        int(df["Area"].max()) + 1,
        10
    )
})

predicted_range = model.predict(area_range)

plt.scatter(
    X["Area"],
    y,
    color="blue",
    label="Actual Data"
)

plt.plot(
    area_range["Area"],
    predicted_range,
    color="red",
    label="SVR Prediction"
)

plt.xlabel("House Area (square feet)")
plt.ylabel("House Price (lakh)")
plt.title("Support Vector Regression")
plt.legend()
plt.grid(True)
plt.show()