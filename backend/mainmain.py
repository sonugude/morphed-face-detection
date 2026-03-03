from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from face_data_collection_app import logic as data_logic
from face_recognition_app import logic as recog_logic
from morph_detection_app import logic as morph_logic

# Import router from face_data_collection_app.logic
from face_data_collection_app.logic import face_data_collection_app
from fastapi.middleware.cors import CORSMiddleware



import uvicorn
import os

app = FastAPI()



# Mount frontend static files (for homepage images, CSS, JS)
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Mount app-specific static folders
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

app.mount(
    "/face_data_collection_static",
    StaticFiles(directory="face_data_collection_app/static"),
    name="face_data_collection_static"
)
app.mount("/face_recognition_app", StaticFiles(directory="face_recognition_app"), name="face_recognition_app")
app.mount("/morph_detection_app", StaticFiles(directory="morph_detection_app"), name="morph_detection_app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# === Template Setup ===

# Templates for homepage and apps
templates_home = Jinja2Templates(directory=frontend_path)
templates_data = Jinja2Templates(directory="face_data_collection_app/templates")
templates_recog = Jinja2Templates(directory="face_recognition_app/templates")
templates_morph = Jinja2Templates(directory="morph_detection_app/templates")
templates = Jinja2Templates(directory=frontend_path)


# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Routers ===
app.include_router(face_data_collection_app, prefix="/face_data_collection")

# === Page Routes ===

@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    return templates_home.TemplateResponse("home page.html", {"request": request})


@app.get("/face_data_collection", response_class=HTMLResponse)
async def face_data_page(request: Request):
    return templates_data.TemplateResponse("index.html", {"request": request})

@app.get("/face_recognition", response_class=HTMLResponse)
async def face_recognition_page(request: Request):
    return templates_recog.TemplateResponse("index2.html", {"request": request})

@app.get("/morph_detection", response_class=HTMLResponse)
async def morph_detection_page(request: Request):
    return templates_morph.TemplateResponse("index3.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates_home.TemplateResponse("about.html", {"request": request})


@app.get("/user-manual", response_class=HTMLResponse)
async def manual(request: Request):
    return templates_home.TemplateResponse("user-manual.html", {"request": request})

# === API Endpoints ===

@app.post("/upload-face/")
async def upload_face(file: UploadFile = File(...)):
    return await data_logic.handle_face_upload(file)

@app.post("/recognize-face/")
async def recognize_face(name: str = Form(...), file: UploadFile = File(...)):
    return await recog_logic.handle_recognition(file, name)

@app.post("/detect-morph/")
async def detect_morph(file: UploadFile = File(...)):
    return await morph_logic.handle_morph(file)




# === Start Server ===
if __name__ == "__main__":
    uvicorn.run("mainmain:app", host="0.0.0.0", port=8000, reload=True)
