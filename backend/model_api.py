import os
import re
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------- Load model(s) from pickle ----------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_models_final.pkl")

print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)

with open(MODEL_PATH, "rb") as f:
    models = pickle.load(f)  # this is a dict like {"svc_final": ..., "logreg_final": ...}

# pick SVC if available, otherwise fallback to logreg
model = models.get("svc_final") or models.get("logreg_final")

# ---------- FastAPI app ----------
app = FastAPI(title="Fake News Classifier API")

# Allow frontend (Cloudflare) to call this API from another domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request schema ----------
class InputData(BaseModel):
    title: str = ""
    text: str = ""

# ---------- Text cleaner (match notebook style) ----------
LINKS_RE = re.compile(r"http\\S+|www\\S+", re.IGNORECASE)
HTML_RE  = re.compile(r"<.*?>", re.IGNORECASE)
SPACE_RE = re.compile(r"\\s+")

def clean_text(s: str) -> str:
    s = s.lower()
    s = LINKS_RE.sub(" ", s)
    s = HTML_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s

# ---------- Routes ----------
@app.get("/")
def root():
    return {"ok": True, "message": "Fake News API is running"}

@app.post("/predict")
def predict(data: InputData):
    # merge title + text same way you did in the notebook
    merged = f"{data.title} {data.text}".strip()
    merged = clean_text(merged)

    # model is a sklearn Pipeline
    pred = model.predict([merged])[0]          # 0 or 1
    label = "REAL" if pred == 0 else "FAKE"

    # crude confidence estimate
    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([merged])[0]
        confidence = float(np.max(proba))
    elif hasattr(model, "decision_function"):
        df = model.decision_function([merged])
        confidence = float(1 / (1 + np.exp(-abs(df[0]))))

    return {
        "ok": True,
        "label": label,
        "label_id": int(pred),
        "confidence": confidence,
    }

# For local testing:
# uvicorn model_api:app --reload --port 8000
