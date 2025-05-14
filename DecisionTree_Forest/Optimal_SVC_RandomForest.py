import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#https://archive.ics.uci.edu/dataset/186/wine+quality
df = pd.read_csv("winequality-white.csv", sep=';')

X = df.drop('quality', axis=1)
y = df['quality']
print(X.columns)
print(df.columns[-1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#RANDOM FOREST
rf = RandomForestClassifier(class_weight='balanced')
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': range(2,15)
}

gs_rf = GridSearchCV(rf, param_grid, cv=5, verbose=3)
gs_rf.fit(X_train_scaled, y_train)

best_params_rf = gs_rf.best_params_

best_rf = RandomForestClassifier(
    n_estimators=best_params_rf['n_estimators'],
    max_depth=best_params_rf['max_depth'],
    random_state=42,
    n_jobs=-1
)

best_rf.fit(X_train_scaled, y_train)

#SVC
svc = SVC(class_weight='balanced', kernel='rbf')
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10]
}

gs_svc = GridSearchCV(svc, param_grid, cv=5, verbose=3)
gs_svc.fit(X_train_scaled, y_train)

best_params_svc = gs_svc.best_params_

best_svc = SVC(
    C=best_params_svc['C'],
    gamma=best_params_svc['gamma'],
    kernel='rbf'
)

best_svc.fit(X_train_scaled, y_train)

#OUTPUT AND VISUALIZATION
score = best_rf.score(X_test_scaled, y_test)
print(f"Random Forest Test Score: {score:.4f}")

score = best_svc.score(X_test_scaled, y_test)
print(f"RBF SVC Test Score: {score:.4f}")

importances = pd.DataFrame(best_rf.feature_importances_, index=df.columns[0:-1])
importances.plot.bar()
plt.show(block=True)

cm1 = confusion_matrix(y_test, best_rf.predict(X_test_scaled))
disp1 = ConfusionMatrixDisplay(cm1, display_labels=np.unique(y_test))
disp1.plot()
plt.show()

cm2 = confusion_matrix(y_test, best_svc.predict(X_test_scaled))
disp2 = ConfusionMatrixDisplay(cm2, display_labels=np.unique(y_test))
disp2.plot()
plt.show()