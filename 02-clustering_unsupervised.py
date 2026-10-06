# Assignment 2:
# Apply Clustering Techniques Using Unsupervised Learning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Create a sample dataset
data = {
    'Annual_Income': [
        15, 18, 20, 22, 25,
        45, 48, 50, 52, 55,
        75, 78, 80, 85, 90
    ],
    'Spending_Score': [
        20, 25, 18, 30, 22,
        45, 50, 42, 55, 48,
        75, 80, 70, 85, 78
    ]
}

df = pd.DataFrame(data)

# Step 2: Select features for clustering
X = df[['Annual_Income', 'Spending_Score']]

# Step 3: Create the K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Step 4: Train the model and assign cluster labels
df['Cluster'] = kmeans.fit_predict(X)

# Step 5: Display the dataset with cluster labels
print("Customer Dataset with Cluster Labels:")
print(df)

# Step 6: Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Step 7: Display the number of data points in each cluster
print("\nNumber of Customers in Each Cluster:")
print(df['Cluster'].value_counts().sort_index())

# Step 8: Visualize the clusters
plt.figure(figsize=(8, 5))

for cluster in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster]

    plt.scatter(
        cluster_data['Annual_Income'],
        cluster_data['Spending_Score'],
        label=f'Cluster {cluster}'
    )

# Display the cluster centers
centers = kmeans.cluster_centers_

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering Using Unsupervised Learning")

plt.legend()
plt.grid(True)
plt.show()