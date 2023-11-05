import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

df = pd.read_csv("Canciones_Spotify.csv")
df = df.drop(columns=["song_title", "artist", "Unnamed: 0", "time_signature", "key", "duration_ms"])
df.head()

Y = df["target"]

def search_best_accuracy(model, df, param_distributions):
    X = df.drop("target", axis=1)
    y = df['target']

    num_columns = X.shape[1]
    best_accuracy = 0
    best_columns_combination = None
    best_model = None

    for i in range(1, num_columns + 1):
        columns_combination = combinations(X.columns, i)

        for combination in columns_combination:
            selected_columns = list(combination)
            Xsub = X[selected_columns]

            X_train, X_test, y_train, y_test = train_test_split(Xsub, y, test_size=0.3, random_state=50)
            param_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5, scoring='accuracy', random_state=42)
            param_search.fit(X_train, y_train)
            y_pred = param_search.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(accuracy, selected_columns)
            print(f"BEST ACCURACY: {best_accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_columns_combination = selected_columns
                best_model = param_search.best_estimator_

    return best_accuracy, best_columns_combination, best_model

search_knn = KNeighborsClassifier()
param_distributions = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
    'leaf_size': [10, 20, 30, 40, 50]
}
best_score, best_combination, best_model = search_best_accuracy(search_knn, df, param_distributions)
with open("knn.txt", "w") as archivo:
    archivo.write(f"Best Score: {best_score}\n")
    archivo.write(f"Columns: {best_combination}\n")
    archivo.write(f"Model: {best_model}")

search_svc = SVC()
param_distributions = {
    'kernel': ['linear', 'rbf', 'sigmoid'],  # Tipo de kernel
}
best_score, best_combination, best_model = search_best_accuracy(search_svc, df, param_distributions)
with open("svc.txt", "w") as archivo:
    archivo.write(f"Best Score: {best_score}\n")
    archivo.write(f"Columns: {best_combination}\n")
    archivo.write(f"Model: {best_model}")


search_dtc = DecisionTreeClassifier()
param_distributions = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None]
}
best_score, best_combination, best_model = search_best_accuracy(search_dtc, df, param_distributions)
with open("dtc.txt", "w") as archivo:
    archivo.write(f"Best Score: {best_score}\n")
    archivo.write(f"Columns: {best_combination}\n")
    archivo.write(f"Model: {best_model}")

search_gnb = GaussianNB()
param_distributions = {
    'var_smoothing': np.logspace(0, -9, num=100)
}
best_score, best_combination, best_model = search_best_accuracy(search_gnb, df, param_distributions)
with open("gnb.txt", "w") as archivo:
    archivo.write(f"Best Score: {best_score}\n")
    archivo.write(f"Columns: {best_combination}\n")
    archivo.write(f"Model: {best_model}")