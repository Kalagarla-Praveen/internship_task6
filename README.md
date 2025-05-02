# 🌸 K-Nearest Neighbors (KNN) Classification on Iris Dataset

## 📌 Objective
To understand and implement the K-Nearest Neighbors (KNN) classification algorithm using the **Iris** dataset. This task includes:
- Data preprocessing
- Feature normalization
- KNN model training and evaluation
- Optimal K value selection
- Visualization of decision boundaries

---

## 🧰 Tools & Libraries
- **Python**
- **Pandas** – Data manipulation
- **Scikit-learn (sklearn)** – Model creation, evaluation
- **Matplotlib** – Data visualization
- **NumPy** – Numerical operations

---

## 📁 Dataset
The dataset used is `Iris.csv`, which includes:
- **Features**: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
- **Target**: Species (Iris-setosa, Iris-versicolor, Iris-virginica)

---

## 🧪 Steps & Operations Performed

### 1. Data Loading
- Loaded the Iris dataset using `pandas`.

### 2. Feature Selection
- Selected only the first two features (`SepalLengthCm`, `SepalWidthCm`) for 2D decision boundary visualization.

### 3. Feature Scaling
- Standardized the selected features using `StandardScaler` to bring them to the same scale.

### 4. Optimal K Selection
- Tried K values from 1 to 10.
- Calculated training accuracy for each K.
- Selected the value with the highest accuracy as **optimal K**.

### 5. KNN Model Training
- Trained the `KNeighborsClassifier` with the selected optimal K using the scaled 2D feature set.

### 6. Decision Boundary Visualization
- Created a mesh grid covering the 2D space.
- Predicted class labels for each point on the grid.
- Mapped string labels to numeric values for visualization.
- Plotted decision boundaries using `matplotlib.contourf`.

---

## 📊 Output
- Printed optimal value of K.
- Displayed scatter plot showing decision boundaries along with actual data points.

---

