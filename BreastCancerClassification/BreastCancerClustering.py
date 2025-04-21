import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
# Load the dataset
# https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
cancer_data = pd.read_csv("wi_breastcancer.csv")

# Extract features (X) and labels (y)
X = cancer_data.iloc[:, 2:]         # Features start from column 2 onward
y = cancer_data.iloc[:, 1]          # Diagnosis column (B or M)

# Encode labels: 'M' (Malignant) -> 1, 'B' (Benign) -> 0
label_encoder = LabelEncoder()
label_encoder.fit(['B', 'M'])
y = label_encoder.transform(y)

# Standardize the feature values to zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering with 2 clusters (Benign or Malignant)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Reduce dimensionality to 2 components using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train a Linear Support Vector Classifier on the PCA-reduced data
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_pca, y)

# Output the accuracy of the SVC on the training data
print(f"Linear SVC Score: {svc.score(X_pca, y):.3f}")

# Create a DataFrame for easy plotting and labeling
df = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'Actual Diagnosis': ['Malignant' if label == 1 else 'Benign' for label in y],
    'Cluster': clusters
})

# Create a mesh grid across the PCA space for plotting SVC decision boundaries
xx, yy = np.meshgrid(
    np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100),
    np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 100)
)

# Predict class labels on the mesh grid for decision boundary visualization
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Set up a 1x2 plot for side-by-side visualizations
plt.figure(figsize=(12, 5))

# Plot KMeans clustering result
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', edgecolor='k', legend='full')
plt.title('KMeans Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Plot actual diagnosis with SVC decision boundaries
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')  # Plot decision boundary
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Actual Diagnosis', palette='Set1', edgecolor='k', legend='full')
plt.title('Actual Diagnosis with SVC')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# Make layout tight and display the plots
plt.tight_layout()
plt.show()

# Align cluster labels to actual labels using majority vote per cluster
labels_fixed = np.zeros_like(clusters)
for cluster_id in np.unique(clusters):
    mask = clusters == cluster_id
    labels_fixed[mask] = mode(y[mask])[0]  # Assign most common actual label in cluster

# Compute normalized confusion matrix between true labels and aligned cluster labels
cm = confusion_matrix(y, labels_fixed, normalize='true')

print("\nConfusion Matrix:")
print(cm)

# Compute and display accuracy, precision, recall for the 'Malignant' class
precision = cm[1, 1]/ (cm[0,1] + cm[1,1])
accuracy = accuracy_score(y, labels_fixed)
print(f"KMeans Accuracy: {accuracy:.3f}")
print(f"KMeans Precision: {precision:.3f}")
print(f"KMeans Recall: {cm[1,1]:.3f}")

# Plot confusion matrix visually
disp = ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Malignant'])
disp.plot()
plt.show()