pip install scikit-learn


from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Create an instance of the Flask class
app = Flask(__name__)

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the dataset
df = pd.read_csv("C:\Users\VIPL2021CLLL161\Downloads\model\DataNeuron_Text_Similarity.csv")  # Replace with the actual filename
text1 = df['text1'].tolist()
text2 = df['text2'].tolist()

# Compute sentence embeddings for text1 and text2
sentence_embeddings1 = model.encode(text1)
sentence_embeddings2 = model.encode(text2)

# Compute cosine similarity between sentence embeddings
similarity_scores = [cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0][0] 
                     for e1, e2 in zip(sentence_embeddings1, sentence_embeddings2)]

# Normalize similarity scores to range [0, 1]
similarity_scores_normalized = [(score + 1) / 2 for score in similarity_scores]

# Add similarity scores to the DataFrame
df['similarity_score'] = similarity_scores_normalized

# Route to serve the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.json
    text1 = data['text1']
    text2 = data['text2']

    # Encode input texts
    sentence_embeddings1 = model.encode(text1)
    sentence_embeddings2 = model.encode(text2)

    # Compute cosine similarity between sentence embeddings
    similarity_score = cosine_similarity(sentence_embeddings1.reshape(1, -1), 
                                          sentence_embeddings2.reshape(1, -1))[0][0]

    # Normalize similarity score to range [0, 1]
    similarity_score_normalized = (similarity_score + 1) / 2

    # Return response with similarity score
    response = {'similarity_score': similarity_score_normalized}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
