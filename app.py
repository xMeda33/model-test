from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
import gc

app = Flask(__name__)

def load_books_data(filepath):
    chunks = []
    chunk_size = 10000  # Adjust the chunk size based on your memory constraints
    for chunk in pd.read_csv(filepath, chunksize=chunk_size, on_bad_lines='skip', encoding='latin-1'):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

# Load book data
final_books = load_books_data('./data/data.csv')

# Combine features for content-based filtering, ensuring no NaN values
final_books.rename(columns={'authors': 'author','categories':'genre','description':'desc','average_rating':'average_rating'}, inplace=True)
final_books['combined_features'] = (final_books['title'].fillna('') + ' ' +
                                    final_books['genre'].fillna('') + ' ' +
                                    final_books['author'].fillna('') + ' ' +
                                    final_books['desc'].fillna(''))

# Initialize CountVectorizer and TfidfTransformer
count_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = count_vectorizer.fit_transform(final_books['combined_features'].fillna(''))
tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

# Placeholder for the collaborative filtering model
model = None

# Load and train collaborative filtering model
def load_and_train_model():
    global model, ratings
    try:
        ratings = pd.read_csv('./data/ratings.csv', on_bad_lines='skip', encoding='latin-1')
        
        # Prepare the dataset for Surprise collaborative filtering
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Train the SVD model
        model = SVD(n_factors=20, biased=True, random_state=42)
        model.fit(trainset)
        print("Model trained successfully")
        
        # Free up memory
        del trainset, testset, data
        gc.collect()
    except Exception as e:
        print(f"Error during model training: {e}")

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
    ratings_file_path = os.path.join(UPLOAD_FOLDER, 'ratings.csv')

    # Check if a file named "ratings" exists and delete it
    if os.path.exists(ratings_file_path):
        os.remove(ratings_file_path)

    try:
        # Save the new file with the name "ratings.csv"
        file.save(ratings_file_path)
        print("File saved successfully")
        
        # Load and train the collaborative filtering model
        load_and_train_model()
    except Exception as e:
        print(f"Error during file upload and model training: {e}")
        return jsonify({"error": "File upload failed"}), 500

    return jsonify({"message": "File uploaded and model trained successfully"}), 200

@app.route('/recommend/content', methods=['GET'])
def recommend_content():
    book_id = int(request.args.get('book_id'))
    recommendations = get_content_recommendations(book_id)
    return jsonify(recommendations.to_dict(orient='records'))

@app.route('/recommend/collaborative', methods=['GET'])
def recommend_collaborative():
    user_id = int(request.args.get('user_id'))
    if model is None:
        return jsonify({"error": "Model not trained. Please upload the ratings file first."}), 400
    user_ratings = ratings[ratings['user_id'] == user_id]
    if len(user_ratings) < 10:
        return jsonify({"message":'This user has rated less than 10 books', "books":recommendations.to_dict(orient='records')})
    recommendations = get_book_recommendations(user_id)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
