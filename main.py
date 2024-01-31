from file_parser import parse_csv
from demographic_filtering import demographic_filtering
from collaborative_filtering import (
    CF_get_top_movies_for_user,
    get_user_item_matrix,
    CF_get_top_movies_for_movie,
)


TMDB_MOVIES = "./data/movies.csv"
TMDB_CREDITS = "./data/credits.csv"
RATINGS_SMALL = "./data/ratings.csv"
# number of first n movies filters will return
RESULT_SIZE = 20


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
    # # DEMOGRAPHIC FILTERING
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
    results = CF_get_top_movies_for_movie("Batman", df_ratings, df_movies, 10)


if __name__ == "__main__":
    main()
