from pydantic import BaseModel
import os
import cv2
import face_recognition
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
import json
from fastapi import APIRouter

# === Path Setup ===
BASE_DIR = Path(__file__).resolve().parent
dataset_dir = BASE_DIR / "dataset"
images_dir = dataset_dir / "images"
encodings_dir = dataset_dir / "encodings"
os.makedirs(images_dir, exist_ok=True)
os.makedirs(encodings_dir, exist_ok=True)

# === Pydantic Model ===
class FrameData(BaseModel):
    name: str
    image: str

# Helper function to create user directories
def create_user_directories(name: str):
    user_img_dir = images_dir / name
    user_enc_dir = encodings_dir / name
    os.makedirs(user_img_dir, exist_ok=True)
    os.makedirs(user_enc_dir, exist_ok=True)
    return user_img_dir, user_enc_dir

# Initialize metadata if not already present
def initialize_metadata(user_enc_dir: Path):
    metadata_path = user_enc_dir / 'metadata.json'
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump({}, f)  # Initialize as empty dictionary
    return metadata_path

# === Core logic for handling frame upload ===
async def handle_upload_frame(data: FrameData):
    name = data.name.strip().lower()
    max_samples = 500
    saved = 0

    try:
        # Decode base64 image data
        image_data = data.image.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)

        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Load face detection cascade
        cascade_path = str(BASE_DIR / "haarcascade_frontalface_default.xml")
        facedetect = cv2.CascadeClassifier(cascade_path)
        faces = facedetect.detectMultiScale(frame_bgr, 1.3, 5)

        if len(faces) == 0:
            return {"message": "❌ No face detected in the frame."}

        # Create directories for the user if not already existing
        user_img_dir, user_enc_dir = create_user_directories(name)

        # Initialize metadata
        metadata_path = initialize_metadata(user_enc_dir)

        # Count existing encodings
        existing_encodings = list(user_enc_dir.glob("*.npy"))
        current_count = len(existing_encodings)

        if current_count >= max_samples:
            return {"message": f"✅ Already collected 500 encodings for {name}."}

        # Process each detected face
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

                    # Save face image and encoding
                    cv2.imwrite(str(img_path), face_img_bgr)
                    np.save(str(enc_path), encodings[0])

                    # Update metadata
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

# === FastAPI Router Setup ===
face_data_collection_app = APIRouter()

@face_data_collection_app.post("/upload-frame/")
async def upload_frame(data: FrameData):
    return await handle_upload_frame(data)
