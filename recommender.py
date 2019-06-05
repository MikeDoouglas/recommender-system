# ################################################################################## #
#            Copyright (c) 2019    #    Mike Douglas Oliveira Coelho                 #
#            All rights reserved.  #    João Henrique Boing Souza                    #
# ################################################################################## #

from copy import copy
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def recommend(title):
    metadata = pandas.read_csv('datasets/movies_metadata_test.csv', low_memory=False)
    metadata['overview'] = metadata['overview'].fillna('')

    tfidf_matrix = create_tfidf_matrix(metadata['overview'])

    indices = pandas.Series(metadata.index, index=metadata['title']).drop_duplicates()
    idx = indices[title]

    cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    similarity_score = list(enumerate(cosine_similarity[idx]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:11]

    movie_indices = [i[0] for i in similarity_score]
    return metadata['title'].iloc[movie_indices]


def create_tfidf_matrix(content):
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(content)


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
