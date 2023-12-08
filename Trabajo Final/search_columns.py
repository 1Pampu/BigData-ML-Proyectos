import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectIMDB/data/movie_metadata.csv")
df = df.dropna()
df = df.select_dtypes(include='number')

Y = df["imdb_score"]

def search_best_data(model, df, param_distributions, file):
    X = df.drop("imdb_score", axis=1)
    y = df['imdb_score']

    num_columns = X.shape[1]
    total_combinations = sum([len(list(combinations(X.columns, i))) for i in range(1, num_columns + 1)])
    current_combination = 0
    
    best_accuracy = -10
    best_columns_combination = None
    best_model = None

    for i in range(1, num_columns + 1):
        columns_combination = combinations(X.columns, i)

        for combination in columns_combination:
            selected_columns = list(combination)
            Xsub = X[selected_columns]

            X_train, _, y_train, _ = train_test_split(Xsub, y, test_size=0.3, random_state=50)
            param_search = RandomizedSearchCV(model,param_distributions,n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
            param_search.fit(X_train, y_train)
            score = param_search.best_score_

            current_combination += 1
            print(f"Combination {current_combination}/{total_combinations}: ", end="")
            print(score, selected_columns)
            print(f"BEST: {best_accuracy}")

            if score > best_accuracy:
                best_accuracy = score
                best_columns_combination = selected_columns
                best_model = param_search.best_estimator_

                with open(f"{file}.txt", "w") as archivo:
                    archivo.write(f"Best Score: {best_accuracy}\n")
                    archivo.write(f"Columns: {best_columns_combination}\n")
                    archivo.write(f"Model: {best_model}")

    return best_accuracy, best_columns_combination, best_model

# search_LR = LinearRegression()
# param_dist = {
#     'fit_intercept': [True, False],
#     'copy_X': [True, False],
#     'n_jobs': [None, -1],
# }
# best_score, best_combination, best_model = search_best_data(search_LR, df, param_dist, "LR")

param_dist_ridge = {
    'alpha': np.logspace(-6, 6, 13),
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
}
search_Ridge = Ridge()
search_best_data(search_Ridge, df, param_dist_ridge, "Ridge Regression")

# Modelo de Regresión Lasso
param_dist_lasso = {
    'alpha': np.logspace(-6, 6, 13),
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'copy_X': [True, False],
}
search_Lasso = Lasso()
search_best_data(search_Lasso, df, param_dist_lasso, "Lasso Regression")

# Modelo de Regresión Elastic Net
param_dist_elasticnet = {
    'alpha': np.logspace(-6, 6, 13),
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'copy_X': [True, False],
}
search_ElasticNet = ElasticNet()
search_best_data(search_ElasticNet, df, param_dist_elasticnet, "Elastic Net Regression")

# Modelo de Regresión de Vectores de Soporte (SVR)
param_dist_svr = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.2, 0.5, 1],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
}
search_SVR = SVR()
search_best_data(search_SVR, df, param_dist_svr, "SVR")

# Modelo de Regresión por K Vecinos Más Cercanos (KNN)
param_dist_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
}
search_KNN = KNeighborsRegressor()
search_best_data(search_KNN, df, param_dist_knn, "KNN")

# Modelo de Árboles de Decisión para Regresión
param_dist_tree = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
search_Tree = DecisionTreeRegressor()
search_best_data(search_Tree, df, param_dist_tree, "Decision Tree")

# Modelo de Bosques Aleatorios para Regresión
param_dist_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
search_RF = RandomForestRegressor()
search_best_data(search_RF, df, param_dist_rf, "Random Forest")

# Modelo de Regresión por Máquinas de Soporte Vectorial con Núcleo (Kernelized SVR)
param_dist_kernelridge = {
    'alpha': np.logspace(-6, 6, 13),
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
}
search_KernelRidge = KernelRidge()
search_best_data(search_KernelRidge, df, param_dist_kernelridge, "Kernel Ridge")

# Modelo de Regresión por Máquinas de Soporte Vectorial con Núcleo Gaussiano (Gaussian Process Regressor)
param_dist_gaussian = {
    'alpha': np.logspace(-6, 6, 13),
}
search_GaussianProcess = GaussianProcessRegressor()
search_best_data(search_GaussianProcess, df, param_dist_gaussian, "Gaussian Process")

# Modelo de Regresión por Gradiente Descendente Estocástico (SGDRegressor)
param_dist_sgd = {
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': np.logspace(-6, 6, 13),
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
}
search_SGD = SGDRegressor()
search_best_data(search_SGD, df, param_dist_sgd, "SGD Regressor")

# Modelo de Regresión por Gradiente Impulso (Gradient Boosting Regressor)
param_dist_gradientboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
search_GradientBoost = GradientBoostingRegressor()
search_best_data(search_GradientBoost, df, param_dist_gradientboost, "Gradient Boosting")

# Modelo de Regresión por Bosques Extra (Extra Trees Regressor)
param_dist_extra = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
search_ExtraTrees = ExtraTreesRegressor()
search_best_data(search_ExtraTrees, df, param_dist_extra, "Extra Trees")

# Modelo de Regresión por Red Neuronal (Multi-layer Perceptron Regressor)
param_dist_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': np.logspace(-6, 6, 13),
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}
search_MLP = MLPRegressor()
search_best_data(search_MLP, df, param_dist_mlp, "MLP Regressor")

# Modelo de Regresión por AdaBoost
param_dist_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'loss': ['linear', 'square', 'exponential'],
}
search_AdaBoost = AdaBoostRegressor()
search_best_data(search_AdaBoost, df, param_dist_adaboost, "AdaBoost Regressor")