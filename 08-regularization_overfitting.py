# Assignment 8: Apply Regularization to Avoid Overfitting

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Load the dataset
diabetes = load_diabetes()

X = pd.DataFrame(
    diabetes.data,
    columns=diabetes.feature_names
)

y = diabetes.target

print("Dataset Loaded Successfully")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 3: Create the models
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0)
    ),

    "Lasso Regression": make_pipeline(
        StandardScaler(),
        Lasso(alpha=1.0, max_iter=10000)
    )
}


# Step 4: Train and evaluate each model
results = []

for name, model in models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Predict training and testing values
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate evaluation metrics
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Store results
    results.append({
        "Model": name,
        "Training MSE": train_mse,
        "Testing MSE": test_mse,
        "Training R2": train_r2,
        "Testing R2": test_r2
    })


# Step 5: Display comparison
results_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(results_df.round(3).to_string(index=False))


# Step 6: Visualize the testing MSE
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["Testing MSE"]
)

plt.xlabel("Regression Models")
plt.ylabel("Testing Mean Squared Error")
plt.title("Comparison of Regularization Techniques")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()


# Step 7: Display the conclusion
print("\nConclusion:")
print(
    "Linear, Ridge, and Lasso Regression models "
    "were trained and evaluated."
)

print(
    "Compare training and testing performance to "
    "understand the effect of regularization."
)