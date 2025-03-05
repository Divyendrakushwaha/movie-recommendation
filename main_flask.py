import os
import pandas as pd
import difflib
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = '/home/divyendrak/mysite/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "supersecretkey"

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Landing Page
@app.route('/')
def index():
    return render_template('index.html')

# Movie Recommendation Route
@app.route('/movie', methods=['GET', 'POST'])
def movie_recommend():
    if request.method == 'POST':
        movie_name = request.form.get('movie')
        if not movie_name:
            flash("Please enter a movie name!", "danger")
            return redirect(url_for('movie_recommend'))

        movies_data = pd.read_csv('movies.csv')
        selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
        movies_data.fillna('', inplace=True)

        combined_features = movies_data[selected_features].agg(' '.join, axis=1)
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)

        similarity = cosine_similarity(feature_vectors)

        list_of_all_titles = movies_data['title'].tolist()
        close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1)

        if not close_match:
            flash("Movie not found!", "danger")
            return redirect(url_for('movie_recommend'))

        index_of_movie = movies_data[movies_data.title == close_match[0]]['index'].values[0]
        similarity_scores = list(enumerate(similarity[index_of_movie]))
        sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommendations = [movies_data.iloc[movie[0]]['title'] for movie in sorted_similar_movies[1:11]]
        print(recommendations)
        return render_template('main.html', my_list=recommendations)

    return render_template('main.html', my_list=[])

# Custom CSV Recommendation Route
@app.route('/upload', methods=['GET', 'POST'])
def custom_recommend():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part!", "danger")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash("No selected file!", "danger")
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            df = pd.read_csv(file_path)
            columns = df.columns.tolist()

            return render_template('select_column.html', filename=filename, columns=columns)

    return render_template('upload.html')

# Process CSV Recommendation
@app.route('/process', methods=['POST'])
def process_csv():
    filename = request.form.get('filename')
    user_query = request.form.get('query')
    selected_columns = request.form.getlist('columns')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        flash("File not found!", "danger")
        return redirect(url_for('custom_recommend'))

    df = pd.read_csv(file_path)
    df = df.astype(str)
    df.fillna('', inplace=True)

    if not selected_columns:
        flash("Please select at least one column!", "danger")
        return redirect(url_for('custom_recommend'))
    print(df.columns)
    df['combined_features'] = df[selected_columns].agg(' '.join, axis=1)
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(df['combined_features'].astype(str))

    query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, feature_vectors).flatten()
    sorted_indices = similarity_scores.argsort()[::-1][:10]

    max_score = similarity_scores.max() if similarity_scores.max() > 0 else 1
    normalized_scores = (similarity_scores[sorted_indices] / max_score) * 100
    normalized_scores = [round(score, 1) for score in normalized_scores]

    print(normalized_scores)
    df = df.drop(columns=['combined_features'])
    
    recommendations = df.iloc[sorted_indices].copy()  # Ensure a copy to avoid SettingWithCopyWarning
    recommendations["score"] = normalized_scores  # Assign scores directly

    # Filter where score > 0
    recommendations = recommendations[recommendations["score"] > 0].to_dict(orient="records")
    if not recommendations:
        flash("Please enter a movie name!", "danger")
    return render_template('recommendation.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run()
