#PARTE 1 - DATA EXPLORATION & PREPROCESSING

#1. Caricamento e analisi dei dati

from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

print(df.head())
print(df.describe())


#2. Visualizzazione con grafici

import matplotlib.pyplot as plt

df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

df.plot(kind = "box", figsize = (12, 6))

from pandas.plotting import scatter_matrix

scatter_matrix(df, figsize = (16, 16))

plt.show()

#3. Preparazione dei dati

from sklearn.model_selection import train_test_split

X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10)


#PARTE 2 - CONFRONTO E SELEZIONE DEI MODELLI

#1. Implementazione modelli
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "SVM": SVR()
}

#2. K-Fold Cross Validation e 3. NMSE
from sklearn.model_selection import cross_val_score

for nome, modello in models.items():
    nmse = cross_val_score(modello, X_train, y_train, scoring= "neg_mean_squared_error")
    
    print(nome, nmse)
    

#PARTE 3 - HYPERPARAMETER TUNING

#1. Grid Search 
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, scoring= "neg_mean_squared_error")

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

#Linear Regression è il miglior modello nel confronto iniziale 
#Alpha ottimale per Ridge trovata tramite Grid Search pari a 0.1
#Ridge ottimizzato ottine un NMSE di -3088, migliore di quello inziale

#2. Valutazione Finale
from sklearn.metrics import mean_squared_error, r2_score

miglior_modello = grid.best_estimator_

y_pred = miglior_modello.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

#PARTE 4 - ANALISI DELLE PERFORMANCE

#. Learning Curve
from sklearn.model_selection import learning_curve
import numpy as np

dimensioni, train, test = learning_curve(miglior_modello, X_train, y_train, scoring= "neg_mean_squared_error")

plt.figure(figsize=(10, 6))
plt.plot(dimensioni, -train.mean(axis=1), label= "Training error")
plt.plot(dimensioni, -test.mean(axis=1), label= "Validation error")

plt.xlabel("Training size")
plt.ylabel("MSE")
plt.title("Learning Curve")
plt.legend()
plt.show()

#Dal grafico possiamo vedere un overfitting.

#PARTE 5 - VISUALIZZAZIONE CON DATA REDUCTION

#1. PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

#2. Plot della Regressione
modello_pca = Ridge(alpha=  grid.best_params_["alpha"])
modello_pca.fit(X_reduced, y)

y_pca = modello_pca.predict(X_reduced)

plt.figure(figsize=(10, 6))
plt.tricontourf(X_reduced[:, 0], X_reduced[:, 1], y_pca, cmap= "viridis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Regression Ridge")
plt.colorbar(label="Target in focus")
plt.show()