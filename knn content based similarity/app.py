from flask import Flask, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load datasets
ratings = pd.read_csv('https://raw.githubusercontent.com/BheemisettySaiHarsha/PRML_project/main/PRML%20DATASET/ratings.csv')
movies = pd.read_csv('https://raw.githubusercontent.com/BheemisettySaiHarsha/PRML_project/main/PRML%20DATASET/movies.csv')
tags = pd.read_csv('https://raw.githubusercontent.com/BheemisettySaiHarsha/PRML_project/main/PRML%20DATASET/tags.csv')
links = pd.read_csv('https://raw.githubusercontent.com/BheemisettySaiHarsha/PRML_project/main/PRML%20DATASET/links.csv')

# Drop timestamp column from tags
tags.drop('timestamp', axis=1, inplace=True)

# Combine tags for each user-movie pair
tags_combined = tags.groupby(['userId', 'movieId'])['tag'].apply(', '.join).reset_index()

# Merge movies with combined tags
dataframe = pd.merge(movies, tags_combined, on='movieId', how='left')

# Convert genres column to strings separated by commas
dataframe['genres'] = dataframe['genres'].apply(lambda x: ','.join(x.split('|')))

# Fill missing tags with empty string
dataframe['tag'] = dataframe['tag'].fillna('')

# Combine tags and genres into a single 'tags' column
dataframe['tags'] = dataframe['tag'] + ', ' + dataframe['genres']

# Drop unnecessary columns
new = dataframe.drop(columns=['tag', 'genres'])

# Vectorize 'tags' column
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vector)

# Define API keys
TMDB_API_KEY = '3b6d898137728d7df661e5ffe4934beb'

# Function to get TMDb poster URL
def get_tmdb_poster(tmdb_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}')
    data = response.json()
    poster_path = data.get('poster_path')
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
    return poster_url

# Function to recommend movies
# Function to recommend movies
def recommend(u_id):
    # Filter movies with ratings greater than or equal to 3 for the given user ID
    high_rated_movies = ratings[(ratings['userId'] == u_id) & (ratings['rating'] >= 3)]

    if not high_rated_movies.empty:
        # Select a random high-rated movie
        movie_id = high_rated_movies.sample(n=1)['movieId'].iloc[0]
    else:
        # If no high-rated movies, select a random movie
        movie_id = ratings['movieId'].sample(n=1).iloc[0]

    # Get the index of the movie in the DataFrame 'new'
    index = new[new['movieId'] == movie_id].index[0]

    # Find similar movies
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    # Create a list to store recommended movie information
    recommendations = []

    # Iterate through the top 5 similar movies
    for i in distances[2:7]:
        similar_movie_id = new.iloc[i[0]]['movieId']
        movie_title = new.iloc[i[0]]['title']
        tmdb_id = links.loc[links['movieId'] == similar_movie_id, 'tmdbId'].values[0]
        tmdb_poster_url = get_tmdb_poster(tmdb_id)
        recommendations.append({'title': movie_title, 'poster_url': tmdb_poster_url})

    return recommendations

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    user_id = int(request.form['user_id'])
    recommendations = recommend(user_id)
    return render_template('recommendation.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
