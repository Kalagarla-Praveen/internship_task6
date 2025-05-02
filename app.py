import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

# Assuming 'df' is your DataFrame containing the Iris dataset
df=pd.read_csv('/content/Iris.csv')
# Extract features and target variable
X = df[['SepalLengthCm', 'SepalWidthCm']]
y = df['Species']

# Normalize the two features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal K based on accuracy (this was missing in previous responses)
k_values = range(1, 11)  # Example range of k values
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)
    y_pred = knn.predict(X_scaled)
    accuracies.append(accuracy_score(y, y_pred))

optimal_k = k_values[accuracies.index(max(accuracies))]
print(f"Optimal K: {optimal_k}")

# Decision boundary using first two features
X_2D = X_scaled[:, :2]
x_min, x_max = X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1
y_min, y_max = X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

knn_2D = KNeighborsClassifier(n_neighbors=optimal_k)
knn_2D.fit(X_2D, y)

Z = knn_2D.predict(np.c_[xx.ravel(), yy.ravel()])
# Convert string labels to numerical values for contourf
Z_numeric = pd.factorize(Z)[0].reshape(xx.shape) #This line was added to convert string labels to numerical values.

plt.figure(figsize=(8, 6))
cmap_background = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_points = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Use Z_numeric instead of Z for contourf
plt.contourf(xx, yy, Z_numeric, cmap=cmap_background, alpha=0.3) # This line was modified to use Z_numeric
plt.scatter(X_2D[:, 0], X_2D[:, 1],
            c=y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}),
            cmap=cmap_points, edgecolor='k')
plt.title(f'Decision Boundaries (First Two Features, K={optimal_k})')
plt.xlabel('SepalLengthCm (standardized)')
plt.ylabel('SepalWidthCm (standardized)')
plt.grid(True)
plt.show()
