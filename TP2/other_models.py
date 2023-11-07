import pandas as pd
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("Canciones_Spotify.csv")
df = df.drop(columns=["song_title", "artist", "Unnamed: 0", "time_signature", "key", "duration_ms"])
df.head()

Y = df["target"]
X = df.drop("target", axis=1)

print("datos cargados")

# Definimos de una función que busca la mejor combinación de columnas para el modelo recibido y con los mejores hiperparametros
def search_best_data(model, df, param_distributions):

    # Obtenemos el número de columnas en el DataFrame
    num_columns = X.shape[1]

    # Inicializamos variables para almacenar la mejor precisión, combinación de columnas y modelo ganador
    best_accuracy = 0
    best_columns_combination = None
    best_model = None

    # Iteramos a través de todas las combinaciones posibles de columnas
    for i in range(1, num_columns + 1):

        # Definimos las combinaciones de columnas posibles con un tamaño "i"
        columns_combination = combinations(X.columns, i)

        for combination in columns_combination:

            # Definimos la lista de columnas en la combinación actual
            selected_columns = list(combination)

            # Definimos el conjunto de columnas a utilizar en esta iteracion y separamos en datos de entrenamiento y prueba
            Xsub = X[selected_columns]
            X_train, _, y_train, _ = train_test_split(Xsub, Y, test_size=0.3, random_state=50)

            # Realizamos una búsqueda aleatoria de hiperparametros para el modelo (Con los parametros recibidos en la funcion)
            param_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5, scoring='accuracy', random_state=42)
            param_search.fit(X_train, y_train)
            score = param_search.best_score_

            print(score, selected_columns)
            print(f"BEST ACCURACY: {best_accuracy}")

            # Actualizamos la mejor precisión y la mejor combinación si encontramos un puntaje mejor
            if score > best_accuracy:
                best_accuracy = score
                best_columns_combination = selected_columns
                best_model = param_search.best_estimator_

    # Devolvemos la mejor precisión, la mejor combinación de columnas y el modelo encontrado
    return best_accuracy, best_columns_combination, best_model


search_rfc = GradientBoostingClassifier()
param_dist = {
    'n_estimators': [10,100,1000],
    'max_depth': [None, 10,100],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
best_score, best_combination, best_model = search_best_data(search_rfc, df, param_dist)
with open("gbc.txt", "w") as archivo:
    archivo.write(f"Best Score: {best_score}\n")
    archivo.write(f"Columns: {best_combination}\n")
    archivo.write(f"Model: {best_model}")
