from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json, numpy as np, os
from datasets import load_dataset
import numpy as np

# === load your saved DistilBERT emotion model ===
load_dir = "./emotion-distilbert-v1"   # adjust if you saved elsewhere

tok = AutoTokenizer.from_pretrained(load_dir)
model = AutoModelForSequenceClassification.from_pretrained(load_dir)
model.config.problem_type = "multi_label_classification"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# === load extra metadata (thresholds + labels) ===
with open(os.path.join(load_dir, "extras.json"), "r", encoding="utf-8") as f:
    extras = json.load(f)

TARGETS   = extras["targets"]
best_tau  = np.array(extras["best_tau"], dtype=np.float32)
print("Loaded model for labels:", TARGETS)

def _sigmoid(x): return 1/(1+np.exp(-x))

def infer(text: str):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0].cpu().numpy()
    probs = _sigmoid(logits)
    preds = (probs >= best_tau).astype(int)
    return probs, preds

te = load_dataset("cardiffnlp/tweet_eval", "emotion")
label_names = te["train"].features["label"].names  # ['anger','joy','optimism','sadness']

# Map TweetEval -> your 7
map7 = {"anger":"anger","joy":"joy","sadness":"sadness","optimism":"joy"}  # or neutral
def to_multi_hot(example):
    y = np.zeros(7, dtype=np.float32)
    name = label_names[example["label"]]
    if map7.get(name):
        idx = ["anger","disgust","fear","joy","sadness","surprise","neutral"].index(map7[name])
        y[idx] = 1.0
    else:
        y[6] = 1.0  # neutral fallback
    example["labels7"] = y.tolist()
    return example
te = te.map(to_multi_hot)