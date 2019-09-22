# ################################################################################## #
#            Copyright (c) 2019    #    Mike Douglas Oliveira Coelho                 #
#            All rights reserved.  #    João Henrique Souza                          #
# ################################################################################## #

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recommend(title):
    metadata = pd.read_csv('datasets/movies_metadata_test.csv', low_memory=False)
    tfidf = TfidfVectorizer(stop_words='english')
    metadata['overview'] = metadata['overview'].fillna('')

    user_movies = ['Toy Story', 'Jumanji']

    # insert new column 'watch' in dataframe
    data_frame_column = 0
    default_value = 0
    metadata.insert(data_frame_column, 'watch', default_value)

    # add value 1 (true) to all movies in 'user_movies'
    metadata.loc[metadata['title'].isin(user_movies), ['watch']] = 1

    tfidf_matrix = tfidf.fit_transform(metadata['overview'])

    # Use formatted_tfidf just to watch on debug
    formatted_tfidf = format_tfidf_results(tfidf, tfidf_matrix, metadata)

    # Agora preciso transformar tfidf_matrix em um formato que de pra fazer o .dot
    # metadata['watch'].dot(tfidf_matrix)

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]


def format_tfidf_results(tfidf, tfidf_matrix, metadata):
    result = {}
    for idx, movies in enumerate(tfidf_matrix):
        movie_title = metadata['title'][idx]
        word_values = movies.data
        word_indices = movies.indices
        words_and_values = {}
        for i, word_index in enumerate(word_indices):
            word_text = tfidf.get_feature_names()[word_index]
            words_and_values[word_text] = word_values[i]
        result[movie_title] = words_and_values
    return result


def main():
    print(recommend('Toy Story'))


if __name__ == '__main__':
    main()
