import numpy as np
from pandas import read_csv


def top_cosine_similarity(data, movie_id, top_n):
    index = movie_id - 1
    movie_row = data[index, :]
    similarity = movie_row @ data.T
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]


def print_similar_movies(movie_data, movie_id, top_indexes):
    print("Recommendations for {movie}: ".format(movie=movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for index in top_indexes + 1:
        print("* {movie}".format(movie=movie_data[movie_data.movie_id == index].title.values[0]))


def main():
    data = read_csv("data/ratings.dat",
                    names=["user_id", "movie_id", "rating", "time"],
                    engine="python", delimiter="::")
    movie_data = read_csv("data/movies.dat",
                           names=["movie_id", "title", "genre"],
                           engine="python", delimiter="::")

    ratings_mat = np.ndarray(shape=(np.max(data.movie_id.values), np.max(data.user_id.values)))
    ratings_mat[data.movie_id.values - 1, data.user_id.values - 1] = data.rating.values

    u, s, vt = np.linalg.svd(ratings_mat.T)

    k = 100
    movie_id = 2628
    top_n = 10

    sliced = vt.T[:, :k]
    indexes = top_cosine_similarity(sliced, movie_id, top_n)
    print_similar_movies(movie_data, movie_id, indexes)


if __name__ == "__main__":
    main()
