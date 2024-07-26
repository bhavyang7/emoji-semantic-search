from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer, util
import emoji
import torch
import logging
import time
from flask_cors import CORS
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Helper function for text normalization
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    return text

# Load emoji data
def get_emoji_data():
    emoji_data = []
    for emoji_char, emoji_info in emoji.EMOJI_DATA.items():
        description = preprocess_text(emoji_info['en'])
        if 'alias' in emoji_info:
            description += ' ' + ' '.join(preprocess_text(alias) for alias in emoji_info['alias'])
        emoji_data.append((emoji_char, description))

    logger.info(f"Loaded {len(emoji_data)} emojis")
    return emoji_data

emoji_data = get_emoji_data()

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can try other models as well
logger.info("Loaded SentenceTransformer model")

# Create embeddings for emoji descriptions
descriptions = [emoji[1] for emoji in emoji_data]
emoji_embeddings = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=False)
logger.info(f"Created embeddings with shape: {emoji_embeddings.shape}")

# Define search function
def search_emoji(query):
    start_time = time.time()
    query = preprocess_text(query)
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    logger.info(f"Encoding query took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    cos_scores = util.cos_sim(query_embedding, emoji_embeddings)[0]
    logger.info(f"Cosine similarity calculation took {time.time() - start_time:.2f} seconds")

    best_result_idx = torch.argmax(cos_scores).item()
    best_score = cos_scores[best_result_idx].item()
    emoji_char, description = emoji_data[best_result_idx]
    return emoji_char, description, best_score

# Define API endpoint
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', default='', type=str)
    if not query:
        return jsonify({'error': 'Query parameter is required'}), 400

    try:
        emoji_char, description, score = search_emoji(query)
        result = {
            'emoji': emoji_char,
            'description': description,
            'score': score
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"An error occurred during search: {e}")
        return jsonify({'error': 'An error occurred during search'}), 500

# Serve the frontend files
@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('frontend', path)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
