# ################################################################################## #
#            Copyright (c) 2019    #    Mike Douglas Oliveira Coelho                 #
#            All rights reserved.  #    João Henrique Souza                          #
# ################################################################################## #

import csv
import datetime
import json
import pandas
import random

from flask import Flask, request

from recommender import recommend, DATASET_PATH


app = Flask(__name__)

 
@app.route('/', methods=['GET'])
def get_top_movies():
  amount = request.args.get('amount')
  movies = pandas.read_csv(DATASET_PATH, low_memory=False)
  ordered_movies = movies.sort_values(['vote_count'], ascending=False)
  ordered_movies = ordered_movies.head(int(amount))
  movies_response = []
  for index, row in ordered_movies.iterrows():
    movie = {
      'id': row['id'],
      'title': row['title'],
      'poster_path': row['poster_path'],
      'overview': row['overview']
    }
    movies_response.append(movie)
  random.shuffle(movies_response)
  return json.dumps(movies_response)


@app.route('/search', methods=['GET'])
def search_movie():
  title = request.args.get('movie-title')
  movies = pandas.read_csv(DATASET_PATH, low_memory=False)
  matched_movies = movies.loc[movies['title'].str.contains(title)]
  movies_response = []
  for index, row in matched_movies.iterrows():
    movie = {
      'id': row['id'],
      'title': row['title'],
      'poster_path': row['poster_path'],
      'overview': row['overview']
    }
    movies_response.append(movie)
  return json.dumps(movies_response)


@app.route('/recommendation', methods=['GET'])
def recommendation():
  movie_title = request.args.get('movie-title')
  movies_dataframe = pandas.read_csv(DATASET_PATH)
  recommendations = recommend(movie_title, movies_dataframe)

  movies_response = []
  for index, row in recommendations.iterrows():
    movie = {
      'id': row['id'],
      'title': row['title'],
      'poster_path': row['poster_path'],
      'overview': row['overview'],
      'is_recommended': True
    }
    movies_response.append(movie)

  random_movies = movies_dataframe.sample(n=5)
  for index, row in random_movies.iterrows():
    movie = {
      'id': row['id'],
      'title': row['title'],
      'poster_path': row['poster_path'],
      'overview': row['overview'],
      'is_recommended': False
    }
    movies_response.append(movie)
  random.shuffle(movies_response)
  return json.dumps(movies_response) 


@app.route('/recommendation-rate', methods=['GET'])
def recommendation_rate():
  user_name = request.args.get('username')
  rating = request.args.get('rating')
  now = datetime.datetime.now()
  dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
  new_csv_row = [user_name, rating, dt_string]

  with open('results/rates.csv', 'a+', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(new_csv_row)

  return json.dumps(new_csv_row)


if __name__ == '__main__':
    app.run(port='3000')
