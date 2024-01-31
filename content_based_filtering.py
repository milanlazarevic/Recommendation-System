from itertools import chain
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    def __init__(self, movies: pd.DataFrame, user_ratings: pd.DataFrame) -> None:
        self.movies = movies.copy()
        self.indices = pd.Series(movies.index, index=movies['title']).drop_duplicates() # map movie title to index
        self.similarity_matrix = None # similarity matrix for all movies

        self.user_ratings = user_ratings.copy()


    # returns cosine similarity matrix based on movie overview text
    def description_similarity(self) -> np.ndarray:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.movies['overview'])
        return cosine_similarity(tfidf_matrix, tfidf_matrix)
    

    # prepare strings for count vectorization by converting to lowercase and space stripping
    def __clean_data(self, x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''
            
    # combine count features into string for vectorization
    def __create_soup(self, x):
        return x['director'] + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['genres']) + ' ' + ' '.join(x['keywords'])


    # returns cosine similarity matrix based on movie genres, keywords, director and top 3 cast members
    def count_similarity(self) -> np.ndarray:
        features = ['cast', 'keywords', 'director', 'genres']
        for feature in features:
            self.movies[feature] = self.movies[feature].apply(lambda x : self.__clean_data(x))
        self.movies['soup'] = self.movies.apply(lambda x : self.__create_soup(x), axis=1)

        count_vectorizer = CountVectorizer(stop_words='english')
        count_matrix = count_vectorizer.fit_transform(self.movies['soup'])
        return cosine_similarity(count_matrix, count_matrix)


    # sets the movie similarity matrix as a linear combination of description and count similarity matrices
    def calculate_scores(self) -> pd.DataFrame:
        # self.similarity_matrix = 0.5 * self.description_similarity() + 0.5 * self.count_similarity()
        self.similarity_matrix = self.description_similarity() * self.count_similarity()
        return self.similarity_matrix

    def __user_movie_similarity(self, idx: int, x) -> float:
        if x['title'] not in self.indices:
            return 0
        if idx > len(self.similarity_matrix) - 1:
            return 0
        if self.indices[x['title']] > len(self.similarity_matrix) - 1:
            return 0
        return x['rating'] * self.similarity_matrix[idx][self.indices[x['title']]]


    # returns weighted sum of similarity scores for movies rated by user and the given movie
    def user_similarity(self, user_id:int, movie_title: str) -> float:
        idx = self.indices[movie_title]
        if type(idx) != np.int64:
            return 0
        user_scores = self.user_ratings[self.user_ratings['userId'] == user_id]
        user_scores = pd.merge(user_scores, self.movies, left_on='movieId', right_on='id')
        user_scores = user_scores[user_scores['title'] != movie_title]
        if user_scores.empty:
            return 0
        return sum(user_scores.apply(lambda x : self.__user_movie_similarity(idx, x), axis=1)) / sum(user_scores['rating'])


    # Returns the dataframe of up to result_size size with the highest similarity scores to the given movie.
    # The return dataframe format is the same as input one, with added score column.
    def get_recommendation_for_movie(self, movie_title: str, result_size: int = 20) -> pd.DataFrame:
        idx = self.indices[movie_title]

        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x : x[1], reverse=True)
        
        if result_size is None:
            result_size = len(similarity_scores)
        # the provided movie is always first
        best_indices = [s[0] for s in similarity_scores[1:result_size+1]]
        result = self.movies.iloc[best_indices].copy()

        # set score column on result
        result.insert(0, "score", [s[1] for s in similarity_scores[1:result_size+1]])
        return result
    
    # Returns the dataframe of up to result_size size with the highest similarity scores for a given user.
    # Similarity scores are calculated based on movies rated by the user.
    # The return dataframe format is the same as input one, with added score column.
    def get_recommendation_for_user(self, user_id: str, result_size: int = 20) -> pd.DataFrame:
        similarity_scores = list(enumerate(self.movies['title'].apply(lambda x : self.user_similarity(user_id, x))))
        similarity_scores = sorted(similarity_scores, key=lambda x : x[1], reverse=True)
        
        if result_size is None:
            result_size = len(similarity_scores)
        best_indices = [s[0] for s in similarity_scores[:result_size]]
        result = self.movies.iloc[best_indices].copy()

        # set score column on result
        result.insert(0, "score", [s[1] for s in similarity_scores[:result_size]])
        result = self.__convert_to_user_ratings(result, user_id)
        return result
    

    # adds user ratings to the given dataframe (converts score column to user_rating column)
    def __convert_to_user_ratings(self, df: pd.DataFrame, user_id: int) -> pd.DataFrame:
        min_score = df['score'].min()
        max_score = df['score'].max()
        df['rate'] = df['score'].apply(lambda x : (x - min_score) / (max_score - min_score) * 5)
        return df