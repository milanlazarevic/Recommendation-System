import pandas as pd
from content_based_filtering import ContentBasedFilter
from collaborative_filtering import CollaborativeFilter
from typing import Literal


class HybridFilter:
    def __init__(self, movies: pd.DataFrame, user_ratings: pd.DataFrame, type: Literal['weighted','match'], alpha:float=0.5) -> None:
        self.movies = movies.copy()
        self.user_ratings = user_ratings.copy()
        self.content_based_filter = ContentBasedFilter(movies, user_ratings)
        self.collaborative_filter = CollaborativeFilter(movies, user_ratings)
        self.type = type
        self.alpha = alpha # weight for content based filter [0,1]


    def fit(self):
        self.content_based_filter.calculate_scores()
        self.collaborative_filter.get_svd_model_ub()
        self.collaborative_filter.get_svd_model_ib()


    def get_recommendation_for_movie(self, movie_title: str, result_size: int = 20) -> pd.DataFrame:
        if self.type == 'weighted':
            df1 = self.content_based_filter.get_recommendation_for_movie(movie_title, None)
            df2 = self.collaborative_filter.get_recommendation_for_movie(movie_title, None)
            return self.__weighted_merge(df1, df2, 'score', result_size)
        elif self.type == 'match':
            df1 = self.content_based_filter.get_recommendation_for_movie(movie_title, result_size)
            df2 = self.collaborative_filter.get_recommendation_for_movie(movie_title, result_size)
            return self.__match_merge(df1, df2, 'score', result_size)


    def get_recommendation_for_user(self, user_id: str, result_size: int = 20) -> pd.DataFrame:
        if self.type == 'weighted':
            df1 = self.content_based_filter.get_recommendation_for_user(user_id, None)
            df2 = self.collaborative_filter.get_recommendation_for_user(user_id, None)
            return self.__weighted_merge(df1, df2, 'score', result_size)
        elif self.type == 'match':
            df1 = self.content_based_filter.get_recommendation_for_user(user_id, result_size)
            df2 = self.collaborative_filter.get_recommendation_for_user(user_id, result_size)
            return self.__match_merge(df1, df2, 'score', result_size)
        

    def __weighted_merge(self, df1: pd.DataFrame, df2: pd.DataFrame, score_col: str, result_size: int) -> pd.DataFrame:
        df1 = df1.rename(columns={score_col: 'score1'})
        df2 = df2.rename(columns={score_col: 'score2'})
        df = pd.merge(df1, df2, on='id')
        df['score'] = self.alpha * df['score1'] + (1 - self.alpha) * df['score2']
        df = df.sort_values(by=['score'], ascending=False)
        return df.head(result_size)
    

    def __match_merge(self, df1: pd.DataFrame, df2: pd.DataFrame, score_col: str, result_size: int) -> pd.DataFrame:
        df1 = df1.rename(columns={score_col: 'score1'})
        df2 = df2.rename(columns={score_col: 'score2'})
        df = pd.merge(df1, df2, on='id')
        df['score'] = max(df['score1'], df['score2'])
        df = df.sort_values(by=['score'], ascending=False)
        return df.head(result_size)