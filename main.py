from hybrid_filtering import HybridFilter
from file_parser import load_movies, parse_csv


def main():
    df = load_movies()
    user_ratings = parse_csv("data/ratings.csv")

    hybrid_filter = HybridFilter(df, user_ratings, 'weighted', 0.5)
    hybrid_filter.fit()

    print(hybrid_filter.get_recommendation_for_movie('The Dark Knight Rises', 10))


if __name__ == "__main__":
    main()
