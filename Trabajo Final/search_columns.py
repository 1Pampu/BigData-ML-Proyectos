import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("https://raw.githubusercontent.com/emmanueliarussi/DataScienceCapstone/master/3_MidtermProjects/ProjectIMDB/data/movie_metadata.csv")
df.drop(columns=["movie_imdb_link", "movie_title", "facenumber_in_poster", "aspect_ratio"], axis = 1, inplace = True)

df.drop("color", axis = 1, inplace = True)
df.drop("plot_keywords", axis = 1, inplace = True)
df.drop("language", axis = 1, inplace = True)

country_count = df["country"].value_counts()
count = country_count[:2].index
df["country"] = df.country.where(df.country.isin(count), "other")

df["content_rating"].fillna("R", inplace = True)

df.dropna(inplace=True)

Y = df["imdb_score"]
X = df.drop(columns=["imdb_score"])

toInt = ["director_name", "actor_2_name", "genres", "actor_1_name", "actor_3_name", "country", "content_rating"]
Label_Encoder = LabelEncoder()

for column in toInt:
    X[column] = Label_Encoder.fit_transform(X[column])

columnas = ['director_name', 'num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',
       'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name',
       'num_voted_users', 'cast_total_facebook_likes', 'actor_3_name',
       'num_user_for_reviews', 'country', 'content_rating', 'budget',
       'title_year', 'actor_2_facebook_likes', 'movie_facebook_likes']

for column in columnas:
    X[column] = np.log(X[column]+1)

def search_best_data(model, param_distributions, file):

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
            print(f"Combination {current_combination}/{total_combinations}")
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
best_score, best_combination, best_model = search_best_data(search_LR, param_dist, "LR")

search_Ridge = Ridge()
param_grid_Ridge = {
    'alpha': [0.1, 1.0, 10.0],
}
best_score_Ridge, best_combination_Ridge, best_model_Ridge = search_best_data(search_Ridge, param_grid_Ridge, "R")

search_RFR = RandomForestRegressor()
param_grid_RFR = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}
best_score_RFR, best_combination_RFR, best_model_RFR = search_best_data(search_RFR, param_grid_RFR, "RFR")