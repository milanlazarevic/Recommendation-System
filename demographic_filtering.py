import pandas as pd
import numpy as np

CORRECT_FACTOR = 100000


def rate_movie(movie, m: float, C: float) -> float:
    v = movie["vote_count"]
    R = movie["vote_average"]
    return ((v / (v + m) * R) + m * (m + v) * C) / CORRECT_FACTOR


# filter the given dataset and returns top resultsize movies
# wr = (v/(v+m)*R) + (m/(m+v)*C)
# v is the number of votes for the movie;
# m is the minimum votes required to be listed in the chart;
# R is the average rating of the movie; And
# C is the mean vote across the whole report
def demographic_filtering(movies: pd.DataFrame, result_size: int = 20) -> pd.DataFrame:
    # v - vote_count
    # R - vote_average
    m = movies["vote_count"].quantile(0.9)
    C = movies["vote_average"].mean()

    # removing all movies with low number of votes
    movies = movies.loc[movies["vote_count"] >= m]
    # rating every movie
    rate = movies.apply(lambda movie: rate_movie(movie, m, C), axis=1)
    movies.insert(0, "rate", rate)
    # sort my resulted movies by rate
    movies = movies.sort_values(by="rate", ascending=False)
    return movies[0:result_size]
