from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import UploadFile
from face_data_collection_app.logic import FrameData, handle_upload_frame

import os
import cv2
import face_recognition
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
import json
from fastapi import FastAPI

app = FastAPI()

@app.post("/upload-frame/")
async def upload_frame(data: FrameData):
    return await handle_upload_frame(data)
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR
templates = Jinja2Templates(directory=str(BACKEND_DIR / "templates"))

# Mount static files
app.mount("/static", StaticFiles(directory=str(BACKEND_DIR / "static")), name="static")

# Dataset directories
dataset_dir = BACKEND_DIR / "dataset"
images_dir = dataset_dir / "images"
encodings_dir = dataset_dir / "encodings"
os.makedirs(images_dir, exist_ok=True)
os.makedirs(encodings_dir, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class FrameData(BaseModel):
    name: str
    image: str

@app.post("/upload-frame/")
async def upload_frame(data: FrameData):
    name = data.name.strip().lower()
    max_samples = 500
    saved = 0

    try:
        # Decode base64 image
        image_data = data.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)

        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Face detection
        cascade_path = str(BACKEND_DIR / "haarcascade_frontalface_default.xml")
        facedetect = cv2.CascadeClassifier(cascade_path)
        faces = facedetect.detectMultiScale(frame_bgr, 1.3, 5)

        if len(faces) == 0:
            return {"message": "❌ No face detected in the frame."}

        # Setup user-specific directories
        user_img_dir = images_dir / name
        user_enc_dir = encodings_dir / name
        os.makedirs(user_img_dir, exist_ok=True)
        os.makedirs(user_enc_dir, exist_ok=True)

        # Count existing encodings
        existing_encodings = list(user_enc_dir.glob("*.npy"))
        current_count = len(existing_encodings)

        if current_count >= max_samples:
            return {"message": f"✅ Already collected 500 encodings for {name}."}

        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            if current_count + saved >= max_samples:
                break

            face_img_bgr = frame_bgr[y:y+h, x:x+w]
            pil_img = Image.fromarray(cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB))
            rgb_face = np.array(pil_img)

            try:
                encodings = face_recognition.face_encodings(rgb_face)
                if encodings:
                    face_index = current_count + saved + 1
                    img_filename = f'{face_index}.jpg'
                    enc_filename = f'{face_index}.npy'
                    img_path = user_img_dir / img_filename
                    enc_path = user_enc_dir / enc_filename
                    metadata_path = user_enc_dir / 'metadata.json'

                    cv2.imwrite(str(img_path), face_img_bgr)
                    np.save(str(enc_path), encodings[0])

                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            try:
                                metadata = json.load(f)
                            except json.JSONDecodeError:
                                metadata = {}

                    metadata[enc_filename] = {
                        'image_filename': img_filename,
                        'face_roi': (int(x), int(y), int(w), int(h))
                    }

                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)

                    print(f"🖼️ Saved {img_filename} and {enc_filename}")
                    saved += 1
                else:
                    print(f"⚠️ No encoding found for face {i+1}")
            except Exception as e:
                print(f"❌ Error during encoding: {e}")
                return {"message": f"❌ Error processing frame: {str(e)}"}

        if saved:
            return {"message": f"✅ Saved {saved} encoding(s) for {name} (Total: {current_count + saved}/{max_samples})."}
        else:
            return {"message": "⚠️ Face detected but encoding failed."}

    except Exception as e:
        return {"message": f"❌ Error processing frame: {str(e)}"}
