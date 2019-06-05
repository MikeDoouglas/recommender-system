# ################################################################################## #
#            Copyright (c) 2019    #    Mike Douglas Oliveira Coelho                 #
#            All rights reserved.  #    João Henrique Boing Souza                    #
# ################################################################################## #

from copy import copy
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def recommend(title):
    dataset = pandas.read_csv('datasets/movies_metadata_test.csv', low_memory=False)
    dataset['overview'] = dataset['overview'].fillna('')

    # Find the movie in dataset by title
    movie = dataset.loc[dataset['title'] == title]

    metadatas = slice_dataset(dataset, 1000)
    results = []
    for metadata in metadatas:
        # Concatenate movie with all metadata slices
        metadata = pandas.concat([metadata, movie])

        tfidf_matrix = create_tfidf_matrix(metadata['overview'])

        cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Get last score similarity because movie was concatenated at the end
        similarity_score = list(enumerate(cosine_similarity[-1]))
        similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        similarity_score = similarity_score[1:11]

        movie_indices = [i[0] for i in similarity_score]
        results.append(metadata['title'].iloc[movie_indices])

    return results


def create_tfidf_matrix(content):
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(content)


def slice_dataset(dataset, slices_length=1000):
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
