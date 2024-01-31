import pandas as pd
import numpy as np

# filter the given dataset and returns top resultsize movies
# wr = (v/(v+m)*R) + (m/(m+v)*C)
# v is the number of votes for the movie;
# m is the minimum votes required to be listed in the chart;
# R is the average rating of the movie; And
# C is the mean vote across the whole report


class DemographicFilter:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.copy()
        self.correct_factor = 100000
        self.m = self.movies["vote_count"].quantile(0.9)
        self.C = self.movies["vote_average"].mean()

    def __rate_movie(self, movie, m: float, C: float) -> float:
        v = movie["vote_count"]
        R = movie["vote_average"]
        return (
            (v / (v + self.m) * R) + self.m * (self.m + v) * self.C
        ) / self.correct_factor

    def get_recommendation(self, result_size: int = 20) -> pd.DataFrame:
        # removing all movies with low number of votes
        movies = self.movies.loc[self.movies["vote_count"] >= self.m]
        # rating every movie
        rate = movies.apply(
            lambda movie: self.__rate_movie(movie, self.m, self.C), axis=1
        )
        movies.insert(0, "rate", rate)
        # sort my resulted movies by rate
        movies = movies.sort_values(by="rate", ascending=False)
        return movies[0:result_size]
