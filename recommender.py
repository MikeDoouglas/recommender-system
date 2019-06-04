# ################################################################################## #
#            Copyright (c) 2019    #    Mike Douglas Oliveira Coelho                 #
#            All rights reserved.  #    João Henrique Boing Souza                    #
# ################################################################################## #

from copy import copy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def recommend(title):
    metadata = pd.read_csv('datasets/movies_metadata.csv', low_memory=False)

    tfidf = TfidfVectorizer(stop_words='english')

    metadata['overview'] = metadata['overview'].fillna('')

    tfidf_matrix = tfidf.fit_transform(metadata['overview'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

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
