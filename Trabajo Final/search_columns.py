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
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectIMDB/data/movie_metadata.csv")
df = df.dropna()

Y = df["imdb_score"]
X = df.drop(columns=["imdb_score", "movie_imdb_link", "movie_title", "aspect_ratio", "language", "facenumber_in_poster", "color"])
toInt = ["director_name", "actor_2_name", "genres", "actor_1_name", "actor_3_name", "plot_keywords", "country", "content_rating"]
Label_Encoder = LabelEncoder()

columnas = ['director_name', 'num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',
       'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name',
       'num_voted_users', 'cast_total_facebook_likes', 'actor_3_name',
       'plot_keywords', 'num_user_for_reviews', 'country', 'content_rating',
       'budget', 'title_year', 'actor_2_facebook_likes',
       'movie_facebook_likes']

for column in toInt:
    X[column] = Label_Encoder.fit_transform(X[column])
    
for column in columnas:
    X[column] = np.log(X[column]+1)



def search_best_data(model, df, param_distributions, file):

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

            X_train, _, y_train, _ = train_test_split(Xsub, Y, test_size=0.3, random_state=50)
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

search_LR = LinearRegression()
param_dist = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'n_jobs': [None, -1],
}
best_score, best_combination, best_model = search_best_data(search_LR, df, param_dist, "LR")
