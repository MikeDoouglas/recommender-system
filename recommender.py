# ################################################################################## #
#            Copyright (c) 2019    #    Mike Douglas Oliveira Coelho                 #
#            All rights reserved.  #    João Henrique Souza                          #
# ################################################################################## #

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


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
    
    # Agora preciso transformar tfidf_matrix em um formato que de pra fazer o .dot
    # metadata['watch'].dot(tfidf_matrix)

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]


def main():
    print(recommend('Toy Story'))


if __name__ == '__main__':
    main()
