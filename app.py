from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

app = Flask(__name__)

# Load data
final_books = pd.read_csv('./data/data.csv',on_bad_lines='skip', encoding='latin-1')
ratings = pd.read_csv('./data/ratings.csv', on_bad_lines='skip', encoding='latin-1')

# Combine features for content-based filtering
final_books['combined_features'] = final_books['title'] + ' ' + final_books['genre'] + ' ' + final_books['author'].fillna('') + ' ' + final_books['desc']

# Initialize CountVectorizer and TfidfTransformer
count_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = count_vectorizer.fit_transform(final_books['combined_features'])
tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

# Prepare the dataset for Surprise collaborative filtering
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train the SVD model
model = SVD(n_factors=20, biased=True, random_state=42)
model.fit(trainset)

# Content-based filtering function
def compute_cosine_similarity(tfidf_matrix, idx, top_n=10):
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
    return similar_indices

def get_content_recommendations(book_id, top_n=10):
    idx = final_books[final_books['book_id'] == book_id].index[0]
    similar_indices = compute_cosine_similarity(tfidf_matrix, idx, top_n)
    return final_books.iloc[similar_indices][['book_id', 'isbn13']]

# Collaborative filtering function
def get_book_recommendations(user_id, top_n=10):
    all_book_ids = final_books['book_id'].unique()
    predictions = [model.predict(user_id, book_id) for book_id in all_book_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_book_ids = [pred.iid for pred in predictions[:top_n]]
    return final_books[final_books['book_id'].isin(top_book_ids)][['book_id', 'isbn13']]

# Define Flask routes
UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Define the path to the "ratings" file
    ratings_file_path = os.path.join(UPLOAD_FOLDER, 'ratings')

    # Check if a file named "ratings" exists and delete it
    if os.path.exists(ratings_file_path):
        os.remove(ratings_file_path)

    # Save the new file with the name "ratings"
    file.save(ratings_file_path)
    return jsonify({"message": "File uploaded successfully"}), 200

@app.route('/recommend/content', methods=['GET'])
def recommend_content():
    book_id = int(request.args.get('book_id'))
    recommendations = get_content_recommendations(book_id)
    return jsonify(recommendations.to_dict(orient='records'))

@app.route('/recommend/collaborative', methods=['GET'])
def recommend_collaborative():
    user_id = int(request.args.get('user_id'))
    user_ratings = ratings[ratings['user_id'] == user_id]
    if len(user_ratings) < 10:
        return jsonify(message='This user has rated less than 10 books')
    recommendations = get_book_recommendations(user_id)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
