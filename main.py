from hybrid_filtering import HybridFilter
from file_parser import load_movies, parse_csv


def main():
    df = load_movies()
    user_ratings = parse_csv("data/ratings.csv")

    hybrid_filter = HybridFilter(df, user_ratings, "weighted", 0.5)
    hybrid_filter.fit()
    results = hybrid_filter.get_recommendation_for_movie("The Dark Knight Rises", 10)
    print(results["title_x"])
    results = hybrid_filter.get_recommendation_for_user(5, 10)
    print(results["title_x"])

if __name__ == "__main__":
    main()
