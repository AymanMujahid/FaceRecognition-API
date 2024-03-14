from flask import Flask, request, jsonify
import os
import base64
from PIL import Image
import numpy as np
import sqlite3
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# CORS(app)
# creating an API object
# api = Api(app)

#prediction api call
# model = joblib.load(open('Model.pkl','rb'))

# model_path = os.environ.get('\FaceRecognition-API/DeepFace_Encoding_DB.ipynb')

# Function to encode image to base64
def img_encoding(img_path):
    with open(img_path, 'rb') as image_file:
        base64_bytes = base64.b64encode(image_file.read())
        base64_string = base64_bytes.decode('utf-8')
    return base64_string

# Function to get face encodings using DeepFace
def get_face_encodings_deepface(image_path):
    encoding = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=True)
    return encoding[0]['embedding']

# Function to insert face encoding into database
def insert_face_encoding(name, encoding):
    conn = sqlite3.connect('face_recognition_25.db')
    c = conn.cursor()
    encoding_str = ','.join(map(str, encoding))
    c.execute("INSERT INTO face_encodings (name, encoding) VALUES (?, ?)", (name, encoding_str))
    conn.commit()
    conn.close()

# Function to fetch all face encodings from database
def fetch_all_encodings(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, encoding FROM face_encodings")
    rows = cursor.fetchall()
    encodings = []
    for row in rows:
        id, name, encoding_str = row
        encoding = np.fromstring(encoding_str, sep=',')
        encodings.append((id, name, encoding))
    conn.close()
    return encodings

# Function to calculate cosine similarity between two face encodings
def calculate_similarity(enco1, enco2):
    enco1_array = np.array(enco1) if not isinstance(enco1, np.ndarray) else enco1
    enco2_array = np.array(enco2) if not isinstance(enco2, np.ndarray) else enco2
    enco1_reshaped = enco1_array.reshape(1, -1)
    enco2_reshaped = enco2_array.reshape(1, -1)
    similarity_score = cosine_similarity(enco1_reshaped, enco2_reshaped)
    return similarity_score

@app.route('/')
def home():
    return 'DeepFace Recognition API V1'

# API endpoint for adding a new face encoding
@app.route('/add_face_encoding', methods=['POST'])
def add_face_encoding():
    data = request.json
    name = data['name']
    img_base64 = data['image']
    img_data = base64.b64decode(img_base64)
    with open('temp_img.jpg', 'wb') as f:
        f.write(img_data)
    encoding = get_face_encodings_deepface('temp_img.jpg')
    insert_face_encoding(name, encoding)
    return jsonify({'message': 'Face encoding added successfully'})

# API endpoint for recognizing a face
@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    data = request.json
    name = data['name']
    img_base64 = data['image']
    img_data = base64.b64decode(img_base64)
    with open('temp_img.jpg', 'wb') as f:
        f.write(img_data)
    encoding = get_face_encodings_deepface('temp_img.jpg')
    all_encodings = fetch_all_encodings('face_recognition_25.db')
    for row in all_encodings:
        _, existing_name, existing_encoding = row
        similarity = calculate_similarity(encoding, existing_encoding)[0][0]
        if similarity >= 0.6:
            return jsonify({'message': f'Face recognized as {existing_name}', 'similarity': similarity, 'embedding': existing_encoding.tolist()})
    return jsonify({'message': 'Face not recognized', 'embedding': []})

if __name__ == '__main__':
    app.run(debug=True)