# Program to identify the association between
# dependent and independent variables

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create a dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Marks': [35, 40, 50, 55, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

# Step 2: Identify independent and dependent variables
X = df['Study_Hours']
Y = df['Marks']

# Step 3: Calculate the correlation coefficient
correlation = X.corr(Y)

# Step 4: Display the dataset
print("Dataset:")
print(df)

# Step 5: Display the correlation coefficient
print("\nCorrelation Coefficient:", round(correlation, 3))

# Step 6: Interpret the relationship
if correlation > 0:
    print("There is a positive linear association.")
elif correlation < 0:
    print("There is a negative linear association.")
else:
    print("There is no linear association.")

# Step 7: Visualize the relationship
plt.scatter(X, Y)

plt.xlabel("Study Hours (Independent Variable)")
plt.ylabel("Marks (Dependent Variable)")
plt.title("Association Between Study Hours and Marks")

plt.grid(True)
plt.show()