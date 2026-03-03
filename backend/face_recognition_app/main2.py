from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile



import face_recognition
import numpy as np
import cv2
import os
import shutil

app = FastAPI()


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODING_PATH = os.path.join(BASE_DIR, '..', 'face_data_collection_app', 'dataset', 'encodings')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})

@app.post("/recognize/")
async def recognize(name: str = Form(...), file: UploadFile = File(...)):
    name = name.strip().lower()
    encoding_dir = os.path.join(ENCODING_PATH, name)

    if not os.path.exists(encoding_dir):
        return {"error": f"No encodings found for user '{name}'."}

    # Load known encodings
    known_encodings = []
    for enc_file in os.listdir(encoding_dir):
        if enc_file.endswith(".npy"):
            encoding = np.load(os.path.join(encoding_dir, enc_file))
            known_encodings.append(encoding)

    # Save uploaded image
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    image = cv2.imread(temp_path)
    os.remove(temp_path)

    if image is None:
        return {"error": "Image could not be read."}

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, face_locations)

    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        if True in matches:
            return {"match": True, "message": f"Match found for '{name}'!"}

    return {"match": False, "message": "No match found."}
