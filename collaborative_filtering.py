from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import cross_validate, KFold, train_test_split
import pandas as pd
import numpy as np
from numpy.linalg import svd

from sklearn.neighbors import NearestNeighbors

from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilter:
    def __init__(self, movies: pd.DataFrame, user_ratings: pd.DataFrame) -> None:
        # rename id column to movieId to match the other dataset
        movies = movies.rename(columns={"id": "movieId"})
        self.movies = movies.copy()
        self.user_ratings = user_ratings.copy()
        self.svd_ub = None
        self.svd_ib = None
        self.knn_model = None
        self.user_item_matrix = None
        self.indices = pd.Series(
            movies.index, index=movies["title"]
        ).drop_duplicates()  # map movie title to index

    def get_svd_model_ub(self) -> None:
        reader = Reader()
        # convert pandas dataframe to surprise dataset
        data = Dataset.load_from_df(
            self.user_ratings[["userId", "movieId", "rating"]], reader
        )

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
        self.user_item_matrix = user_item_matrix

    # Resize movie dataset to get most popular movies to apply collaborativve filtering on them
    def __resize_movie_dataset(self, movie_df: pd.DataFrame, procentage: float = 0.7):
        m = movie_df["vote_count"].quantile(procentage)
        movie_df = movie_df.loc[movie_df["vote_count"] >= m]
        return movie_df

    def __get_idx_from_title(self, title: str):
        idx = self.indices[title]
        return idx

    def __get_movie_id(self, movie_name: str):
        print(self.movies.columns)
        print(self.movies["title"].unique())
        movie = self.movies[self.movies["title"] == movie_name]
        # movie = self.movies.index[self.movies["title"] == movie_name]
        # if we have more than one movie with the same name but they are different
        # i will just send the first because later we will send id with each movie!
        if len(movie) > 1:
            # returns the first movie id
            print(movie.iloc[0]["movieId"])
            return movie.iloc[0]["movieId"]

        elif movie.empty:
            return None
        try:
            print(movie.iloc[0]["movieId"])
            movie_id = int(movie.iloc[0]["movieId"])
            return movie_id
        except ValueError:
            return None

    def get_recommendation_for_movie(
        self, movie_name: str, result_size: int = 20
    ) -> pd.DataFrame:
        movie_id = self.__get_movie_id(movie_name)
        if movie_id is None:
            print("Movie doesnt exist")
            return None
        model = self.svd_ib
        rate = self.movies.apply(
            lambda movie: model.predict(movie_id, movie["movieId"])[3], axis=1
        )
        # copy movies dataframe to not change the original one
        movies = self.movies.copy()
        movies.insert(0, "score", rate)
        movies = movies.sort_values(by="score", ascending=False)
        # rename movieId to id to match the other dataset
        movies = movies.rename(columns={"movieId": "id"})
        return movies[0:result_size]

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
        movies.insert(0, "score", rate)
        # print(movies.head(5))
        movies = movies.sort_values(by="score", ascending=False)
        # rename movieId to id to match the other dataset
        movies = movies.rename(columns={"movieId": "id"})
        return movies[0:result_size]

    def get_knn_model(self, result_size: int = 10):
        knn = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=result_size, n_jobs=-1
        )
        knn.fit(self.user_item_matrix.transpose())
        self.knn_model = knn

    def get_recommendation_for_movie_knn(self, movie_name: str, result_size: int = 10):
        movie_id = self.__get_movie_id(movie_name)
        # movie_id = self.indices[movie_name]
        if movie_id is None:
            print("Movie doesnt exist")
            return None
        model = self.knn_model
        # print(self.user_item_matrix.columns)
        if ("rating", movie_id) not in self.user_item_matrix.columns:
            print("Movie doesnt exist")
            return None
        movie = self.user_item_matrix[("rating", movie_id)]
        # print(movie)
        # print(self.user_item_matrix.shape)
        distances, indices = model.kneighbors(
            self.user_item_matrix[("rating", movie_id)].values.reshape(1, -1),
            n_neighbors=30,
        )
        # print(distances)
        # print(indices)

        # find the movies with the same id as the indices
        idx_dist_pairs = zip(indices[0], distances[0])
        idx_dist_dict = {}
        for pair in idx_dist_pairs:
            idx_dist_dict[pair[0]] = pair[1]
        result = self.movies[self.movies["movieId"].isin(indices[0])]
        result["score"] = result["movieId"].apply(lambda x: idx_dist_dict[x])
        result = result.sort_values(by="score", ascending=False)
        return result


# cf = CollaborativeFilter(df, user_ratings)
#     cf.get_user_item_matrix()
#     cf.get_knn_model()
#     df = df.dropna()
#     print(df.head())
#     print(cf.user_item_matrix)
#     results = cf.get_recommendation_for_movie_knn("American Pie", 10)
#     if results is not None:
#         print(
#             results[
#                 ["movieId", "title", "vote_average", "vote_count", "genres", "score"]
#             ]
#         )
