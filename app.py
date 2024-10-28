from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load the movies data
movies_data = pd.read_csv('G:/Movie Recommendation System/movies.csv')


# Selecting relevant features for the recommendation system
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replacing the null values with an empty string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combining all the selected features into a single string
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Converting the text data to feature vectors using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate the cosine similarity between the feature vectors
similarity = cosine_similarity(feature_vectors)

# Route to display the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the recommendation request
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        recommended_movies = [movies_data.iloc[movie[0]]['title'] for movie in sorted_similar_movies[1:31]]
        return render_template('index.html', movie_name=close_match, recommended_movies=recommended_movies)
    else:
        return render_template('index.html', error="No close match found. Please try another movie name.")

if __name__ == '__main__':
    app.run(debug=True)
