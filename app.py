#Ensemble API endpoint

import pandas as pd
import numpy as np
import requests
from flask import Flask, request, jsonify
from lightfm import LightFM
from lightfm.data import Dataset as LightFMDataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
import os
from flask_cors import CORS

# Initialize the Flask app
app = Flask(_name_)
CORS(app, resources={r"*": {"origins": "*"}})

os.environ['KAGGLE_USERNAME'] = 'xmeda42069'
os.environ['KAGGLE_KEY'] = '3229046ca04c8a0af470e95562c51db7'

# Function to download Kaggle dataset
def download_kaggle_dataset(dataset_owner, dataset_name, download_path='data'):
    os.makedirs(download_path, exist_ok=True)
    os.system(f'kaggle datasets download -d {dataset_owner}/{dataset_name} -p {download_path}')
    os.system(f'unzip -o {download_path}/{dataset_name}.zip -d {download_path}')


# Load data
download_kaggle_dataset('xmeda42069', 'books-grad-project-please-work')

final_books = pd.read_csv('./data/70k_books.csv', encoding='latin-1', on_bad_lines='skip')
ratings = pd.read_csv('./data/ratings.csv', encoding='latin-1', on_bad_lines='skip')

# Prepare the data for LightFM
lightfm_dataset = LightFMDataset()
lightfm_dataset.fit((x for x in ratings['user_id'].unique()), (x for x in ratings['book_id'].unique()))

# Build the interaction matrix
(interactions, weights) = lightfm_dataset.build_interactions(((row['user_id'], row['book_id']) for _, row in ratings.iterrows()))

# Initialize and train the LightFM model
model = LightFM(loss='warp')
model.fit(interactions, epochs=30, num_threads=2)

# Create reverse mappings
user_id_mapping = lightfm_dataset._user_id_mapping
item_id_mapping = lightfm_dataset._item_id_mapping
reverse_item_id_mapping = {v: k for k, v in item_id_mapping.items()}

# Content-based filtering

final_books['combined_features'] = (final_books['title'].fillna('') + ' ' +
                                    final_books['genre'].fillna('') + ' ' +
                                    final_books['author'].fillna('') + ' ' +
                                    final_books['desc'].fillna(''))
count_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = count_vectorizer.fit_transform(final_books['combined_features'])
tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)


@app.route('/fetch-ratings', methods=['POST'])
def fetch_ratings():
    data = request.json
    
    if not all(key in data for key in ['user_id', 'book_id', 'rating']):
        return jsonify({"error": "Invalid request data. Required fields: user_id, book_id, rating"}), 400
    
    new_row = pd.DataFrame(data, index=[0])
    
    global ratings
    ratings = pd.concat([ratings, new_row], ignore_index=True)
    
    # Write the updated ratings dataframe to the CSV file
    ratings.to_csv('./data/ratings.csv', index=False, encoding='latin-1')
    
    return jsonify({"message": "Rating added successfully"}), 201
def compute_cosine_similarity(tfidf_matrix, idx, top_n=10):
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
    similar_scores = cosine_similarities[similar_indices]
    return similar_indices, similar_scores

def get_content_recommendations(book_id, top_n=10):
    idx = final_books[final_books['book_id'] == book_id].index[0]
    similar_indices = compute_cosine_similarity(tfidf_matrix, idx, top_n)
    return final_books.iloc[similar_indices][['book_id', 'isbn13']]


def get_user_recommendations(user_id, ratings_df, books_df, tfidf_matrix, rating_threshold=3.5, top_n=10):
    user_rated_books = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] > rating_threshold)]
    if user_rated_books.empty:
        top_rated_books = books_df.nlargest(10000, 'rating').sample(n=top_n, random_state=42)
        return top_rated_books[['book_id', 'isbn']]
    
    all_similar_books = []
    for _, row in user_rated_books.iterrows():
        book_id = row['book_id']
        idx = books_df[books_df['book_id'] == book_id].index[0]
        similar_indices, similar_scores = compute_cosine_similarity(tfidf_matrix, idx, top_n)
        for i in range(len(similar_indices)):
            all_similar_books.append((similar_indices[i], similar_scores[i]))
    
    similar_books_scores = {}
    for idx, score in all_similar_books:
        if idx in similar_books_scores:
            similar_books_scores[idx] += score
        else:
            similar_books_scores[idx] = score

    sorted_similar_books = sorted(similar_books_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_indices = [idx for idx, score in sorted_similar_books[:top_n]]
    return books_df.iloc[recommended_indices][['book_id', 'isbn']]

def get_top_n_item_recommendations(user_id, model, books_df, lightfm_dataset, n=5):
    user_id_internal = user_id_mapping[user_id]
    scores = model.predict(user_id_internal, np.arange(lightfm_dataset.interactions_shape()[1]))
    top_n_item_ids = np.argsort(-scores)[:n]
    top_n_books = [reverse_item_id_mapping[item_id] for item_id in top_n_item_ids]
    top_n_books_df = books_df[books_df['book_id'].isin(top_n_books)]
    return top_n_books_df[['book_id', 'isbn']]

def get_ensemble_recommendations(user_id, model, ratings_df, books_df, tfidf_matrix, top_n=10, weight_content=0.6, weight_collaborative=0.4):
    cf_recommendations = get_top_n_item_recommendations(user_id, model, books_df, lightfm_dataset, n=top_n)
    cb_recommendations = get_user_recommendations(user_id, ratings_df, books_df, tfidf_matrix, top_n=top_n)
    cf_recommendations['score'] = weight_collaborative
    cb_recommendations['score'] = weight_content
    all_recommendations = pd.concat([cf_recommendations, cb_recommendations])
    all_recommendations = all_recommendations.groupby('book_id').agg({'score': 'sum'}).reset_index()
    recommended_indices = all_recommendations.nlargest(top_n, 'score')['book_id']
    return books_df[books_df['book_id'].isin(recommended_indices)][['book_id', 'isbn']]

@app.route('/recommend/content', methods=['GET'])
def recommend_content():
    book_id = int(request.args.get('book_id'))
    recommendations = get_content_recommendations(book_id)
    response =  jsonify(recommendations.to_dict(orient='records'))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    recommendations = get_ensemble_recommendations(user_id, model, ratings, final_books, tfidf_matrix)
    return jsonify(recommendations.to_dict(orient='records'))

if _name_ == '_main_':
    app.run(debug=True)