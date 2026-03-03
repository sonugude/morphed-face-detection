from fastapi import UploadFile
from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np
import face_recognition
import cv2
import os

# Setup model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "morph_detector.pth")
device = torch.device("cpu")

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except FileNotFoundError:
    raise FileNotFoundError(f"Model not found at {model_path}")

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def detect_morph(face_image):
    pil_image = Image.fromarray(face_image).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    return "Real" if pred.item() == 1 else "Morphed"

async def handle_morph(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(image_rgb)
    if not face_locations:
        return {"error": "No face detected in the uploaded image."}

    results = []
    for top, right, bottom, left in face_locations:
        face = image_rgb[top:bottom, left:right]
        morph_result = detect_morph(face)
        results.append({"location": (top, right, bottom, left), "result": morph_result})

    return {"faces_detected": len(results), "results": results}
