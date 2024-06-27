import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model

# Initialize the Flask app
app = Flask(__name__)
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

# Content-based filtering setup
final_books['combined_features'] = (final_books['title'].fillna('') + ' ' +
                                    final_books['genre'].fillna('') + ' ' +
                                    final_books['author'].fillna('') + ' ' +
                                    final_books['desc'].fillna(''))
count_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = count_vectorizer.fit_transform(final_books['combined_features'])
tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

# Collaborative filtering using TensorFlow
class CollaborativeFilteringModel:
    def __init__(self, num_users, num_items, embedding_size=50):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.build_model()

    def build_model(self):
        # User and item input layers
        user_input = Input(shape=(1,))
        item_input = Input(shape=(1,))

        # User and item embedding layers
        user_embedding = Embedding(self.num_users, self.embedding_size)(user_input)
        item_embedding = Embedding(self.num_items, self.embedding_size)(item_input)

        # Flatten embeddings
        user_flat = Flatten()(user_embedding)
        item_flat = Flatten()(item_embedding)

        # Dot product of user and item embeddings
        prediction = Dot(axes=1)([user_flat, item_flat])

        # Model creation
        self.model = Model(inputs=[user_input, item_input], outputs=prediction)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, user_ids, item_ids):
        return self.model.predict([user_ids, item_ids])

# Prepare data for collaborative filtering model
user_ids = ratings['user_id'].unique()
item_ids = ratings['book_id'].unique()
num_users = len(user_ids)
num_items = len(item_ids)

# Mapping user and item IDs to sequential integers for model input
user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

ratings['user_index'] = ratings['user_id'].map(user_id_to_index)
ratings['item_index'] = ratings['book_id'].map(item_id_to_index)

X_train = [ratings['user_index'].values, ratings['item_index'].values]
y_train = ratings['rating'].values

# Initialize and train collaborative filtering model
cf_model = CollaborativeFilteringModel(num_users, num_items, embedding_size=50)
cf_model.train(X_train, y_train, epochs=10, batch_size=32)

# Define functions for recommendations
def get_content_recommendations(book_id, top_n=10):
    idx = final_books[final_books['book_id'] == book_id].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
    return final_books.iloc[similar_indices][['book_id', 'isbn13']]

def get_collaborative_recommendations(user_id, ratings_df, books_df, top_n=10):
    user_index = user_id_to_index.get(user_id)
    if user_index is None:
        return pd.DataFrame(columns=['book_id', 'isbn'])  # User not found
    
    user_ids = np.array([user_index] * len(item_ids))
    item_ids = np.array(list(item_id_to_index.values()))
    predictions = cf_model.predict(user_ids, item_ids)
    top_indices = np.argsort(-predictions.flatten())[:top_n]
    top_book_ids = [item_ids[idx] for idx in top_indices]
    return books_df[books_df['book_id'].isin(top_book_ids)][['book_id', 'isbn']]

def get_ensemble_recommendations(user_id, ratings_df, books_df, top_n=10):
    content_recommendations = get_content_recommendations(user_id, top_n=top_n)
    collaborative_recommendations = get_collaborative_recommendations(user_id, ratings_df, books_df, top_n=top_n)
    ensemble_recommendations = pd.concat([content_recommendations, collaborative_recommendations])
    ensemble_recommendations = ensemble_recommendations.drop_duplicates(subset=['book_id'])
    return ensemble_recommendations[['book_id', 'isbn']]

# Flask API endpoints
@app.route('/recommend/content', methods=['GET'])
def recommend_content():
    book_id = int(request.args.get('book_id'))
    recommendations = get_content_recommendations(book_id)
    return jsonify(recommendations.to_dict(orient='records'))

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    recommendations = get_ensemble_recommendations(user_id, ratings, final_books)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
