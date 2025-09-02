import os
import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/sentiment_model.pkl")
VECT_PATH = os.getenv("VECT_PATH", "models/vectorizer.pkl")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

app = FastAPI(title="Sentiment API")

# Allow CORS
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates + static
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class InputText(BaseModel):
    text: str

model = None
vectorizer = None

@app.on_event("startup")
def load_artifacts():
    global model, vectorizer
    print("Loading model from:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    print("Model loaded ✅")

# UI route
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API health check
@app.get("/health")
def health():
    return {"status": "ok"}

# API predict (for external apps/postman)
@app.post("/predict")
def predict(input: InputText):
    vec = vectorizer.transform([input.text])
    pred = model.predict(vec)[0]
    return {"text": input.text, "sentiment": str(pred)}

# API predict (for frontend UI calls)
@app.post("/api/predict")
def predict_ui(input: InputText):
    vec = vectorizer.transform([input.text])
    pred = model.predict(vec)[0]
    return {"text": input.text, "sentiment": str(pred)}
