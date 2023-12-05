import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
import os, shutil, gdown, zipfile
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Crear directorio para almacenar los datos descargados
download_dir = "downloaded_data"
os.makedirs(download_dir, exist_ok = True)

url = "https://drive.google.com/uc?export=download&id=19ZgvHREcc3SGYST_Nczl5XiHHgwvFPbp"
output = download_dir + '/dataset-xray.zip'
gdown.download(url, output , quiet = False)
print("Descarga Completa!")

print("Extrayendo archivos...")
with zipfile.ZipFile(output, 'r') as zip_ref:
    # Extrae todos los archivos en el directorio de destino
    zip_ref.extractall(download_dir)
print("Datos extraidos!")

# Definir las categorías
categories = ["COVID-19", "NEUMONIA", "NORMAL"]

# Cargar datos de las carpetas extraídas
data = []
label = []

print("Cargando datos...")
for category in categories:
    print(f"Cargando radiografias de {category}...")
    for file in os.listdir(download_dir + "/" + category):
        dir = download_dir + "/" + category + "/" + file
        img = imread(dir, as_gray = True)
        img = resize(img, (256, 256))
        data.append(img.flatten())
        label.append(categories.index(category))
print("Todos los datos cargados!")

print("Eliminando el directorio descargado...")
shutil.rmtree(download_dir)
print("Directorio eliminado.")

data = np.asarray(data)
label = np.asarray(label)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, shuffle = True, stratify = label, random_state = 42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_gb = GradientBoostingClassifier(random_state = 42)

# Creamos un objeto GridSearch para la busqueda de parametros y ajustamos a los datos de entrenamiento
grid_search = GridSearchCV(grid_gb, param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)

# Obtenemos los mejores hiperparametros encontrados
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Imprimimos el mejor modelo junto a su puntuacion
print("Model:", best_params)
print("Best Score:", best_score)

with open("gb.txt", "w") as archivo:
    archivo.write(f"Best Score: {best_score}\n")
    archivo.write(f"Parameters: {best_params}\n")


param_grid = {
    'hidden_layer_sizes': [(50,), (100, 50), (100, 100)],
    'activation': ['logistic', 'tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [200, 400, 600]
}

grid_mlp = MLPClassifier(random_state = 42)

# Creamos un objeto GridSearch para la busqueda de parametros y ajustamos a los datos de entrenamiento
grid_search = GridSearchCV(grid_mlp, param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)

# Obtenemos los mejores hiperparametros encontrados
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Imprimimos el mejor modelo junto a su puntuacion
print("Model:", best_params)
print("Best Score:", best_score)

with open("MLP.txt", "w") as archivo:
    archivo.write(f"Best Score: {best_score}\n")
    archivo.write(f"Parameters: {best_params}\n")