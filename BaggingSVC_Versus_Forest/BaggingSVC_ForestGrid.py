import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("weather_classification_data.csv").sample(8000)
encoder = LabelEncoder()
df = df.drop(['Weather Type', 'Season'], axis=1)

X = df.iloc[:, 0:-1].copy()
y = df.iloc[:, -1].copy()

print(X.columns)
print(df.columns[-1])

scaler = StandardScaler()
X = scaler.fit_transform(X)


#RANDOM FOREST WITH GRID SEARCH
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [1,2,3,4,5,6,7,8,9],
}

rf = RandomForestClassifier(random_state=42, oob_score=True)
gs_rf = GridSearchCV(rf, rf_param_grid, cv=5, n_jobs=-1, verbose=3)
gs_rf.fit(X, y)

rf_best_params = gs_rf.best_params_

best_rf = RandomForestClassifier(
    n_estimators=rf_best_params['n_estimators'],
    max_depth=rf_best_params['max_depth'],
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
best_rf.fit(X, y)
best_rf_pred = best_rf.predict(X)

#LINEAR SVC BAGGING CLASSIFIER
svc = SVC(kernel='linear', class_weight='balanced')

param_grid = {
    'C': [0.1, 0.2, 0.3, .4]
}

gs_svc = GridSearchCV(svc, param_grid, cv=5, verbose=3)
gs_svc.fit(X, y)

best_params_svc = gs_svc.best_params_

best_svc = SVC(
    C=best_params_svc['C'],
    kernel='linear',
    class_weight='balanced'
)

bag = BaggingClassifier(
    estimator=svc,
    random_state=42,
    n_estimators=50,
    bootstrap=True,
    oob_score=True,
    verbose=3
)
bag.fit(X, y)
bag_pred = bag.predict(X)


#RANDOM FOREST SCORES
print("Random Forest with Grid Search")
print(f"Score (Train): {best_rf.score(X, y):.3f}")
print(f"OOB Score: {best_rf.oob_score:.3f}")

cv_scores = cross_val_score(best_rf, X,y, cv=5)
print(f"Forest Cross-validation scores: {cv_scores}")

#BAGGING SCORES
print("\nBagging Classifier with Linear SVC")
print(f"Score (Train): {bag.score(X, y):.3f}")
print(f"OOB Score: {bag.oob_score_:.3f}\n")

#RANDOM FOREST CONFUSION MATRIX
cm = confusion_matrix(y, best_rf_pred, normalize='true')
print("Random Forest Confusion Matrix")
print(cm)
print("\n")
disp1 = ConfusionMatrixDisplay(cm, display_labels=best_rf.classes_)
disp1.plot()
plt.show()

#RANDOM FOREST IMPORTANCES
importances = pd.DataFrame(best_rf.feature_importances_, index=df.columns[0:-1])
importances.plot.bar()
plt.show(block=True)

#BAGGING CONFUSION MATRIX
cm2 = confusion_matrix(y, bag_pred, normalize='true')
print("Bagging Confusion Matrix")
print(cm2)
disp2 = ConfusionMatrixDisplay(cm2, display_labels=bag.classes_)
disp2.plot()
plt.show(block=True)

