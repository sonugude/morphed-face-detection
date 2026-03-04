from fastapi import UploadFile, Form, File
import face_recognition
import numpy as np
import cv2
import os
import shutil

# Adjust the base path to point to face_data_collection_app's dataset/encodings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODING_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'face_data_collection_app', 'dataset', 'encodings'))

async def handle_recognition(file: UploadFile, name: str):
    name = name.strip().lower()
    encoding_dir = os.path.join(ENCODING_PATH, name)

    if not os.path.exists(encoding_dir):
        return {"error": f"❌ No encodings found for user '{name}'."}

    known_encodings = []
    for enc_file in os.listdir(encoding_dir):
        if enc_file.endswith(".npy"):
            encoding = np.load(os.path.join(encoding_dir, enc_file))
            known_encodings.append(encoding)

    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    image = cv2.imread(temp_path)
    os.remove(temp_path)

    if image is None:
        return {"error": "❌ Image could not be read."}

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, face_locations)

    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        if True in matches:
            return {"match": True, "message": f"✅ Match found for '{name}'!"}

    return {"match": False, "message": "❌ No match found."}
