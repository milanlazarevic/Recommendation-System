import pandas as pd
import numpy as np
from ast import literal_eval
from typing import List


MOVIES_FILE = 'data/movies.csv'
CREDITS_FILE = 'data/credits.csv'
RATINGS_FILE = 'data/ratings.csv'
MOVIE_ATTRS = [
        "id",
        "title",
        "genres",
        "budget",
        "keywords",
        "vote_average",
        "vote_count",
        "tagline",
        "popularity",
        "release_date",
        "runtime",
        "overview",
    ]
RATINGS_ATTRS = ['userId', 'movieId', 'rating']


# Reads given CSV file and returns a pandas dataframe
def parse_csv(filepath: str, columns: list = None) -> pd.DataFrame:
    if columns is None:
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath, usecols=columns)
    return df


def load_movies(columns: List[str] = None) -> pd.DataFrame:
    # load movies data
    df = parse_csv(MOVIES_FILE, MOVIE_ATTRS)

    # merge credits data to movies dataframe
    df_credits = parse_csv(CREDITS_FILE)
    df_credits.columns = ['id','tittle','cast','crew']
    df = df.merge(df_credits, on='id')

    # convert missing string values to empty strings
    df['overview'] = df['overview'].fillna('')
    df = df.drop_duplicates(subset=['id', 'title'])
    df = df.dropna()

    to_numeric_cols = ['id', 'budget', 'popularity']
    df[to_numeric_cols] = df[to_numeric_cols].apply(pd.to_numeric, errors='coerce')

    to_object_cols = ['cast', 'crew', 'keywords', 'genres']
    for col in to_object_cols:
        df = df.dropna(subset=[col])
        df[col] = df[col].apply(literal_eval)

    # extract data from objects
    df['director'] = df['crew'].apply(get_director)

    names_cols = ['cast', 'keywords', 'genres']
    for col in names_cols:
        df[col] = df[col].apply(get_top3_names)

    if columns is not None:
        df = df[columns]
    return df


# extract director's name from crew
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# returns the first 3 names from a list
def get_top3_names(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


def load_ratings() -> pd.DataFrame:
    df = parse_csv(RATINGS_FILE, RATINGS_ATTRS)
    df = df.dropna()
    df['rating'] = df['rating'].apply(pd.to_numeric, errors='coerce')
    return df
