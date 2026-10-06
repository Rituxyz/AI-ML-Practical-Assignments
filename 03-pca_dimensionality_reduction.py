# Assignment 3:
# Apply Dimensionality Reduction Using PCA

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load the Wine dataset
wine = load_wine()

X = wine.data
y = wine.target

# Step 2: Display the original dataset information
print("Original Dataset Information")
print("----------------------------")
print("Number of samples:", X.shape[0])
print("Number of original features:", X.shape[1])

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# Step 5: Display the transformed dataset
pca_df = pd.DataFrame(
    X_pca,
    columns=["Principal Component 1", "Principal Component 2"]
)

print("\nTransformed Dataset:")
print(pca_df.head())

# Step 6: Display explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

total_variance = sum(pca.explained_variance_ratio_) * 100

print(
    "\nTotal Variance Retained:",
    round(total_variance, 2),
    "%"
)

# Step 7: Visualize the principal components
plt.figure(figsize=(9, 6))

for class_id, class_name in enumerate(wine.target_names):
    plt.scatter(
        X_pca[y == class_id, 0],
        X_pca[y == class_id, 1],
        label=class_name
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA: Dimensionality Reduction of Wine Dataset")

plt.legend(title="Wine Class")
plt.grid(True)

plt.show()