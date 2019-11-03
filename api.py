import csv
import pandas

from flask import Flask, request
from recommender import recommend, DATASET_PATH


app = Flask(__name__)

 
@app.route('/', methods=['GET'])
def get_top_movies():
  amount = request.args.get('amount')
  movies = pandas.read_csv(DATASET_PATH, low_memory=False)
  ordered_movies = movies.sort_values(['vote_count'], ascending=False)
  ordered_movies = ordered_movies.head(amount)
  return ordered_movies.to_json()


@app.route('/search', methods=['GET'])
def search_movie():
  title = request.args.get('movie-title')
  movies = pandas.read_csv(DATASET_PATH, low_memory=False)
  movie = movies.loc[movies['title'] == title]
  return movie.to_json()


@app.route('/recommendation', methods=['GET'])
def recommendation():
  movie_title = request.args.get('movie-title')
  recommendations = recommend(movie_title)
  return recommendations.to_json()


# @app.route('/recommendation-rating', methods=['POST'])
# def recommendation_rating():
#   rating = request.form.get('rating')
#   return 'recommendation rating endpoint'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='3000')
