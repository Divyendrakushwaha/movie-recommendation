from flask import Flask
from flask import request, render_template

import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
 
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/movie',methods=['GET', 'POST'])
def my_route():

    movie_name = request.form.get('movie')
    print(movie_name)
    my_list = []
    if movie_name != None:
        movies_data = pd.read_csv('movies.csv')
        print(movies_data.head())

        selected_features = ['genres','keywords','tagline','cast','director']
        print(selected_features)

        for feature in selected_features:
            movies_data[feature] = movies_data[feature].fillna('')

        combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)

        similarity = cosine_similarity(feature_vectors)
        print(similarity)

        print(similarity.shape)

        # tell your movie name
        # movie_name = "now you see me"

        list_of_all_titles = movies_data['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        print(find_close_match)

        close_match = find_close_match[0]
        print(close_match)

        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        print(index_of_the_movie)

        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        print(similarity_score)

        sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
        # print(sorted_similar_movies)

        print('Movies suggested for you : \n')

        i = 1
        
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index==index]['title'].values[0]
            if (i<30):
                print(i, '.',title_from_index)
                i+=1
                my_list.append(title_from_index)
        return render_template('main.html', my_list=my_list)
    else:
        return render_template('main.html')
 
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()