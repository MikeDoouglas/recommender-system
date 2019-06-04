# ################################################################################## #
#            Copyright (c) 2019    #    Mike Douglas Oliveira Coelho                 #
#            All rights reserved.  #    João Henrique Souza                          #
# ################################################################################## #

from copy import copy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def recommend(title):
    # Read CSV
    metadata = pd.read_csv('datasets/movies_metadata.csv', low_memory=False)

    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    metadata['overview'] = metadata['overview'].fillna('')

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(metadata['overview'])

    # tfidf_matrix.shape = Output (102972, 156266)
    # That means 156,266 different words were used to describe the 102,972 games in our dataset.

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Get the index of the movie that matches the title
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]


def slice_dataset(dataset, slices_length=10000):
    sliced_dataset = []
    while not dataset.empty:
        if len(dataset) < slices_length:
            sliced_dataset.append(dataset[0:len(dataset)])
            dataset = dataset[len(dataset):-1]
        else:
            sliced_dataset.append(dataset[0:slices_length])
            dataset = dataset[slices_length:-1]
    return sliced_dataset


def main():
    print(recommend('Radio'))


if __name__ == '__main__':
    main()
