from file_parser import parse_csv
from demographic_filtering import DemographicFilter
from collaborative_filtering import CollaborativeFilter


TMDB_MOVIES = "./data/movies.csv"
TMDB_CREDITS = "./data/credits.csv"
RATINGS_SMALL = "./data/ratings.csv"
# number of first n movies filters will return
RESULT_SIZE = 5


def main():
    attributes = ["movie_id", "title", "cast", "crew"]
    df_credits = parse_csv(TMDB_CREDITS, attributes)
    # changing the name of cols
    df_credits.columns = ["id", "title", "cast", "crew"]

    # # df movies
    attributes = [
        "genres",
        "id",
        "keywords",
        "vote_average",
        "vote_count",
        "tagline",
        "popularity",
        "release_date",
        "runtime",
        "overview",
    ]
    df_movies = parse_csv(TMDB_MOVIES, attributes)

    df_movies = df_movies.merge(df_credits, on="id")
    # DEMOGRAPHIC FILTERING
    recommendations = DemographicFilter(df_movies).get_recommendation(RESULT_SIZE)
    print(recommendations[["rate", "title", "vote_count", "vote_average"]])

    # results = demographic_filtering(df_movies, RESULT_SIZE)

    # print(results[["rate", "title", "vote_count", "vote_average"]])

    # COLLABORATIVE FILTERING
    attributes = ["userId", "movieId", "rating"]
    # loading the files
    df_movies = df_movies.rename(columns={"id": "movieId"})
    # print(df_movies["movieId"])
    df_ratings = parse_csv(RATINGS_SMALL, attributes)
    # print(df_ratings.head(4))
    titles = df_movies[["title", "movieId"]]
    df_ratings_title = df_ratings.merge(titles, on="movieId")
    # print(df_ratings_title.head(4))
    # print(get_user_item_matrix(df_ratings_title))
    # results = CF_get_top_movies_for_user(1, df_ratings, df_movies, RESULT_SIZE)
    # print(results[["rate", "title", "vote_count", "vote_average"]])
    # GET RECOMMENDATIONS W COLLABORATIVE FILTERING FOR MOVIE NAME
    cf = CollaborativeFilter(df_movies, df_ratings)
    cf.get_svd_model_ub()
    results = cf.get_recommendation_for_user(1, RESULT_SIZE)
    print(results[["rate", "title", "vote_count", "vote_average"]])


if __name__ == "__main__":
    main()
