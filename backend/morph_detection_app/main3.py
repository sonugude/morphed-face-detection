from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import models, transforms
from fastapi import UploadFile

import torch
import numpy as np
import face_recognition
import cv2
import os
import io

app = FastAPI()

# Serve static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Determine the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the model path relative to the script directory
model_path = os.path.join(script_dir, "model", "morph_detector.pth")
device = torch.device("cpu")

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    import sys
    sys.exit(1)
model.to(device)
model.eval()

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Morph detection function
def detect_morph(face_image):
    pil_image = Image.fromarray(face_image).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    return "Morphed" if pred.item() == 1 else "Real"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index3.html", {"request": request})

@app.post("/detect_morph/")
async def detect_morph_api(file: UploadFile = File(...)):
    if not file:
        return JSONResponse({"error": "No file uploaded"}, status_code=400)

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # ✅ Fixed here
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(image_rgb)

        if not face_locations:
            return JSONResponse({"error": "No face detected in the uploaded image."}, status_code=400)

        results = []
        for top, right, bottom, left in face_locations:
            face = image_rgb[top:bottom, left:right]
            morph_result = detect_morph(face)
            results.append({"location": (top, right, bottom, left), "result": morph_result})

        return JSONResponse({"faces_detected": len(results), "results": results})

    except Exception as e:
        return JSONResponse({"error": f"Error processing the image: {str(e)}"}, status_code=500)
