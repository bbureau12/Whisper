from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np, json, os

MODEL_DIR = "./emotion-distilbert-v1"

# ---- load model, tokenizer, thresholds ----
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.config.problem_type = "multi_label_classification"

with open(os.path.join(MODEL_DIR, "extras.json"), "r", encoding="utf-8") as f:
    extras = json.load(f)
TARGETS = extras["targets"]
TAU = np.array(extras["best_tau"], dtype=np.float32)
VERSION = os.path.basename(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def sigmoid(x): return 1/(1+np.exp(-x))

# ---- FastAPI setup ----
app = FastAPI(title="Astraea Emotion API", version=VERSION)

class In(BaseModel):
    text: str
    lang: str | None = "en"
    source: str | None = None
    ts: str | None = None

@app.get("/health")
def health():
    return {"ok": True, "version": VERSION, "device": str(device)}

@app.post("/predict")
def predict(inp: In):
    enc = tok(inp.text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0].detach().cpu().numpy()
    probs = sigmoid(logits)
    active_mask = probs >= TAU
    scores = {name: float(p) for name, p in zip(TARGETS, probs)}
    active = [name for name, m in zip(TARGETS, active_mask) if m]
    top_i = int(np.argmax(probs))
    top = {"label": TARGETS[top_i], "score": float(probs[top_i])}

    # simple flags
    low_conf = float(probs[top_i]) < 0.5
    multi_label = len(active) > 1

    return {
        "version": VERSION,
        "targets": TARGETS,
        "scores": scores,
        "active": active,
        "top": top,
        "flags": {"low_confidence": low_conf, "multi_label": multi_label, "possible_sarcasm": False},
    }
