from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import cross_validate, KFold, train_test_split
import pandas as pd
import numpy as np
from numpy.linalg import svd


def get_svd_model(df: pd.DataFrame) -> svd:
    reader = Reader()
    data = Dataset.load_from_df(df, reader)
    kf = KFold()
    svd = SVD()
    results = cross_validate(svd, data)
    print(results)
    trainset, testset = train_test_split(data, test_size=0.25)
    svd.fit(trainset)
    predictions = svd.test(testset)
    # Then compute RMSE
    accuracy.rmse(predictions)
    # print(svd.predict(1, 302, 3))
    return svd


def CF_get_top_movies_for_user(
    user_id: int,
    df_users: pd.DataFrame,
    df_movies: pd.DataFrame,
    result_size: int = 20,
):
    movies = resize_movie_dataset(df_movies, 0.9)
    model = get_svd_model(df_users)
    print(movies.head(5))
    print(movies.columns)
    # rating every movie
    rate = movies.apply(
        lambda movie: model.predict(user_id, movie["movieId"])[3], axis=1
    )
    movies.insert(0, "rate", rate)
    print(movies.head(5))
    movies = movies.sort_values(by="rate", ascending=False)
    return movies[0:result_size]


def CF_get_top_movies_for_movie(
    movieName: str,
    df_users: pd.DataFrame,
    df_movies: pd.DataFrame,
    result_size: int = 20,
):
    movie_id = get_movie_id(movieName, df_movies)
    print(movie_id)


def get_movie_id(movieName: str, movies: pd.DataFrame):
    movie = movies.loc[movies["title"] == movieName]
    # if we have more than one movie with the same name but they are different
    # i will just send the first because later we will send id with each movie!
    if len(movie) > 1:
        return movie.iloc[0]["movieId"]
    return movie["movieId"]


# Resize movie dataset to get most popular movies to apply collaborativve filtering on them
def resize_movie_dataset(movie_df: pd.DataFrame, procentage: float = 0.7):
    m = movie_df["vote_count"].quantile(procentage)
    movie_df = movie_df.loc[movie_df["vote_count"] >= m]
    return movie_df


def get_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    user_item_matrix = df.pivot(
        index=["userId"], columns=["movieId"], values=["rating"]
    ).fillna(0)
    return user_item_matrix
