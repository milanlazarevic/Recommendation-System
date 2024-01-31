from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import cross_validate, KFold, train_test_split
import pandas as pd
import numpy as np
from numpy.linalg import svd


class CollaborativeFilter:
    def __init__(self, movies: pd.DataFrame, user_ratings: pd.DataFrame) -> None:
        self.movies = movies.copy()
        self.user_ratings = user_ratings.copy()
        self.svd_ub = None
        self.svd_ib = None

    def get_svd_model_ub(self) -> None:
        reader = Reader()
        # convert pandas dataframe to surprise dataset
        data = Dataset.load_from_df(self.user_ratings, reader)

        svd = SVD()
        # Perform train-test split
        trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

        svd.fit(trainset)

        predictions = svd.test(testset)

        # Then compute RMSE
        accuracy.rmse(predictions)

        self.svd_ub = svd

    def get_svd_model_ib(self) -> None:
        reader = Reader()
        # convert pandas dataframe to surprise dataset
        data = Dataset.load_from_df(
            self.user_ratings[["movieId", "movieId", "rating"]], reader
        )

        svd = SVD()
        # Perform train-test split
        trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

        svd.fit(trainset)

        predictions = svd.test(testset)

        # Then compute RMSE
        accuracy.rmse(predictions)

        self.svd_ib = svd

    def get_user_item_matrix(self) -> pd.DataFrame:
        user_item_matrix = self.user_ratings.pivot(
            index=["userId"], columns=["movieId"], values=["rating"]
        ).fillna(0)
        return user_item_matrix

    # Resize movie dataset to get most popular movies to apply collaborativve filtering on them
    def __resize_movie_dataset(self, movie_df: pd.DataFrame, procentage: float = 0.7):
        m = movie_df["vote_count"].quantile(procentage)
        movie_df = movie_df.loc[movie_df["vote_count"] >= m]
        return movie_df

    def __get_movie_id(self, movie_name: str):
        movie = self.movies.loc[self.movies["title"] == movie_name]
        # if we have more than one movie with the same name but they are different
        # i will just send the first because later we will send id with each movie!
        if len(movie) > 1:
            print(movie.iloc[0]["movieId"])
            return movie.iloc[0]["movieId"]

        elif movie.empty:
            return None
        try:
            movie_id = int(movie["movieId"])
            return movie_id
        except ValueError:
            return None

    def get_recommendation_for_movie(
        self, movie_name: str, result_size: int = 20
    ) -> pd.DataFrame:
        movie_id = self.__get_movie_id(movie_name, df_movies)
        if movie_id is None:
            print("Movie doesnt exist")
            return None
        model = self.svd_ib
        rate = df_movies.apply(
            lambda movie: model.predict(movie_id, movie["movieId"])[3], axis=1
        )
        df_movies.insert(0, "rate", rate)
        df_movies = df_movies.sort_values(by="rate", ascending=False)
        return df_movies[0:result_size]

    def get_recommendation_for_user(
        self, user_id: str, result_size: int = 20
    ) -> pd.DataFrame:
        # movies = resize_movie_dataset(df_movies, 0.9)
        movies = self.movies
        model = self.svd_ub
        # print(movies.head(5))
        # print(movies.columns)
        # rating every movie
        rate = movies.apply(
            lambda movie: model.predict(user_id, movie["movieId"])[3], axis=1
        )
        movies.insert(0, "rate", rate)
        # print(movies.head(5))
        movies = movies.sort_values(by="rate", ascending=False)
        return movies[0:result_size]
