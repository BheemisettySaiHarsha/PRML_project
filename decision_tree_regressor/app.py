from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved model from file
with open('decision_tree_regressor.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the movies dataset
movies_df = pd.read_csv('movies.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_id = int(request.form['user_id'])
    all_movie_ids = movies_df['movieId'].unique()
    movies_to_recommend = [movie_id for movie_id in all_movie_ids]

    # Make predictions for each movie
    user_predictions = loaded_model.predict([[user_id, movie_id] for movie_id in movies_to_recommend])

    # Sort movies based on predicted ratings (in descending order)
    recommended_movies = sorted(zip(movies_to_recommend, user_predictions), key=lambda x: x[1], reverse=True)

    # Get top 10 recommendations with movie titles
    top_recommendations = []
    for movie_id, predicted_rating in recommended_movies[:10]:
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        top_recommendations.append((movie_title, predicted_rating))

    return render_template('recommendations.html', user_id=user_id, recommendations=top_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
