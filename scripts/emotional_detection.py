from datasets import load_dataset, Sequence, Value, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy
from sklearn.metrics import f1_score
import numpy as np, torch

# --- 0) Load + show schema
ds = load_dataset("go_emotions", "simplified")
print("feature keys:", list(ds["train"].features.keys())[:60])
print("row0 keys   :", list(ds["train"][0].keys())[:60])

# --- 1) Define target set + source groups
TARGETS = ["anger","disgust","fear","joy","sadness","surprise","neutral"]
GROUPS = {
    "anger":   {"anger","annoyance","disapproval","remorse"},
    "disgust": {"disgust"},
    "fear":    {"fear","nervousness"},
    "joy":     {"joy","amusement","approval","excitement","gratitude","love","optimism","relief","pride","admiration","caring","desire"},
    "sadness": {"sadness","disappointment","embarrassment","grief"},
    "surprise":{"surprise","realization","curiosity","confusion"},
    "neutral": {"neutral"},
}
ALL_28 = {
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral",
}

# --- 2) Detect schema & build mapper (works for all common variants)
features = ds["train"].features
all_keys = set(features.keys())

# Case A: per-emotion columns exist
present_emotions = sorted(list(all_keys & ALL_28))
has_28_cols = len(present_emotions) >= 20

# Case B: multi-label list of ids
has_labels_list = "labels" in features and isinstance(features["labels"], Sequence) and isinstance(features["labels"].feature, ClassLabel)

# Case C: single class id
has_label_single = "label" in features and isinstance(features["label"], ClassLabel)

if has_28_cols:
    print(f"Detected schema: 28 columns ({len(present_emotions)} present).")
    def map_to_targets(example):
        y = np.zeros(len(TARGETS), dtype=np.float32)
        for i, t in enumerate(TARGETS):
            for src in GROUPS[t]:
                # counts or 0/1; any >0 counts as chosen
                if int(example.get(src, 0)) > 0:
                    y[i] = 1.0
                    break
        example["multi_hot"] = y.tolist()
        return example

elif has_labels_list:
    print("Detected schema: labels = Sequence(ClassLabel)")
    src_names = features["labels"].feature.names
    def map_to_targets(example):
        y = np.zeros(len(TARGETS), dtype=np.float32)
        chosen = {src_names[i] for i in example["labels"]}
        for i, t in enumerate(TARGETS):
            if GROUPS[t] & chosen:
                y[i] = 1.0
        example["multi_hot"] = y.tolist()
        return example

elif has_label_single:
    print("Detected schema: label = ClassLabel (single label)")
    src_names = features["label"].names
    def map_to_targets(example):
        y = np.zeros(len(TARGETS), dtype=np.float32)
        chosen = {src_names[example["label"]]}
        for i, t in enumerate(TARGETS):
            if GROUPS[t] & chosen:
                y[i] = 1.0
        example["multi_hot"] = y.tolist()
        return example

else:
    raise RuntimeError(f"Unrecognized schema. Feature keys: {sorted(list(all_keys))[:60]}")

# --- 3) Map and sanity-check BEFORE tokenization
ds = ds.map(map_to_targets)
ds = ds.cast_column("multi_hot", Sequence(Value("float32")))

def stats(split):
    arr = np.asarray(ds[split]["multi_hot"], dtype=np.float32)
    print(split, "rows:", len(arr))
    print(" per-class positives:", arr.sum(axis=0).astype(int).tolist())
    print(" avg labels/row:", float(arr.sum()) / len(arr))
stats("train"); stats("validation")

# Optional: also show raw source totals if we have 28-cols
if has_28_cols:
    def source_totals(split="train", n=5000):
        sample = ds[split].select(range(min(n, len(ds[split]))))
        totals = {c: int(np.asarray(sample[c]).sum()) for c in present_emotions}
        nz = {k:v for k,v in totals.items() if v > 0}
        print(f"{split} source nonzero emotions ({len(nz)}):", sorted(nz.items(), key=lambda x:-x[1])[:8])
    source_totals("train"); source_totals("validation")

# --- 4) Tokenize + attach labels
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(batch):
    enc = tok(batch["text"], truncation=True, max_length=128)
    enc["labels"] = batch["multi_hot"]
    return enc

keep = ("input_ids","attention_mask","labels")
ds = ds.map(tokenize, batched=True).remove_columns([c for c in ds["train"].column_names if c not in keep])

# enforce float32 label tensors (avoid Float→Long issues)
ds = ds.cast_column("labels", Sequence(Value("float32")))
ds.set_format(type="torch", columns=list(keep))

# --- 5) Model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(TARGETS),
)
model.config.problem_type = "multi_label_classification"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- 6) Metrics
def _sigmoid(x): return 1/(1+np.exp(-x))
def compute_metrics(ep):
    logits = getattr(ep, "predictions", None)
    labels = getattr(ep, "label_ids", None)
    if logits is None: logits, labels = ep
    probs = _sigmoid(logits)
    preds = (probs >= 0.5).astype(int)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0)
    }

# --- 7) Training (Windows-safe)
args = TrainingArguments(
    output_dir="./emotion-poc",
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    dataloader_num_workers=0,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    compute_metrics=compute_metrics, processing_class=tok  # future-proof vs tokenizer deprecation
)

# sanity: dtype/shape
b = next(iter(trainer.get_train_dataloader()))
print("labels tensor:", b["labels"].dtype, b["labels"].shape)  # torch.float32, (B, 7)

trainer.train()

# --- 8) Threshold tuning
pred = trainer.predict(ds["validation"])
val_logits = getattr(pred, "predictions", pred[0])
val_labels = getattr(pred, "label_ids", pred[1])
val_probs  = _sigmoid(val_logits)

best_tau = []
for i in range(len(TARGETS)):
    taus = np.linspace(0.1, 0.9, 17)
    yi = val_labels[:, i]
    best_t, best_f1 = 0.5, 0.0
    for t in taus:
        f1 = f1_score(yi, (val_probs[:, i] >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    best_tau.append(best_t)

# --- 9) Inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

def infer(text: str):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0].detach().cpu().numpy()
    probs = _sigmoid(logits)
    preds = (probs >= np.array(best_tau)).astype(int)
    res = [{"label": n, "score": float(p)} for n, p in zip(TARGETS, probs)]
    top = max(res, key=lambda r: r["score"])
    return {"top": top, "results": res, "preds": preds.tolist()}

samples = [
    "I’m so happy we shipped today!",
    "This is infuriating. The build broke again.",
    "I’m worried the deadline is impossible.",
    "I can’t stop crying about it.",
    "Whoa, that came out of nowhere!",
    "That’s disgusting. I feel sick.",
    "Meeting moved to 3pm. See you then.",
    "Yeah, fantastic job… totally not broken 🙃",
]
for s in samples:
    out = infer(s)
    print("\nTEXT:", s)
    print("TOP :", out["top"])
    print("ALL :", [(r["label"], round(r["score"], 3)) for r in out["results"]])
