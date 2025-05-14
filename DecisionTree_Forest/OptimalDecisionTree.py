import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Hotel Reservations.csv")

encode = LabelEncoder()

categorical_columns = df.columns[5:8]
for col in categorical_columns:
    df[col] = encode.fit_transform(df[col])

X = df.iloc[:, 1:9].copy()
print(X.columns)
X = X.to_numpy()
y = df.iloc[:, -1].copy()
print(df.columns[-1])
y = y.to_numpy()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Perform grid search for optimal max_depth
dt = DecisionTreeClassifier()
parameters = {"max_depth": range(2, 15)}

grid_search = GridSearchCV(dt, param_grid=parameters)
grid_search.fit(X_train, y_train)

# Train the decision tree with the optimal depth
best_depth = grid_search.best_params_["max_depth"]
print(f"Optimal max_depth: {best_depth}")

clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42, class_weight="balanced")

# Cross-validation
cv_scores = cross_val_score(clf, X,y, cv=5)
print(f"Cross-validation scores: {cv_scores}")

clf.fit(X_train, y_train)
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=["Canceled", "Not Canceled"])
disp_cm.plot()

plt.figure(figsize=(35, 35))
plot_tree(clf, max_depth=3, feature_names=df.columns[1:9], class_names=["Canceled", "Not Canceled"], filled=True)
plt.show()
