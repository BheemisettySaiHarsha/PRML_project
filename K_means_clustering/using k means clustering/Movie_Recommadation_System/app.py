from flask import Flask, render_template, request
import pandas as pd
import pickle
import random

app = Flask(__name__)

def load_dataset():
    file = open('new_dataframe_pickle.pkl', 'rb')
    new_data = pickle.load(file)
    file.close()
    return new_data

#This function select the cluster for a user according the the user choice
def select_cluster(movie_id):
    new_data = load_dataset()
    cluster_no = new_data.loc[new_data['movieId'] == movie_id, 'Cluster'].values[0]
    return cluster_no

# Load the recommendations for the selected cluster
def get_recommendations(cluster_no):
    new_data = load_dataset()
    recommendations = new_data.loc[new_data['Cluster'] == cluster_no, 'title'].sample(n=20).tolist()
    return recommendations

# Home page route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("Request method:", request.method)  # Print request method for debugging
        movie_id = int(request.form['movie_id'])
        cluster = select_cluster(movie_id)
        recommendations = get_recommendations(cluster)
        return render_template('index.html', prediction_text=recommendations)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
