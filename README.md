# Face Recognition API

RESTful API for face recognition built using Flask and DeepFace.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AymanMujahid/face-recognition-api.git

## Usage
## Adding a Face Encoding
Endpoint: /add_face_encoding (POST)

## Request   
{
  "name": "name",
  
  "image": "base64_encoded_image_data"
}

## Response
{
  "message": "Face encoding added successfully"
}

## Recognizing a Face
Endpoint: /recognize_face (POST)

## Request
{
  "name": "name",
  
  "image": "base64_encoded_image_data"
}

## Response
{
  "message": "Face recognized as name",
  
  "similarity": 0.85,
  
  "embedding": [0.1, 0.2, ..., 0.9]
}

///////////////

[the clone URL](https://github.com/AymanMujahid/face-recognition-api.git), make sure to customize the content according to Face Recognition API functionality and requirements.

