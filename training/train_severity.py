"""
training/train_severity.py
==========================
Milestone 5: fine-tune DistilBERT with LoRA to classify weather-text severity.

Run from project root:  python training/train_severity.py

Compares three systems on the held-out test set:
  1. Untrained baseline  — DistilBERT with a randomly initialised head (≈ chance)
  2. Keyword heuristic   — hand-written rules (a strong non-ML baseline)
  3. LoRA fine-tuned     — what we ship

Saves the LoRA adapter + classifier head + label map to models/severity_classifier/.
"""

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.severity_data import LABELS, build_dataset

BASE_MODEL = "distilbert-base-uncased"
OUT_DIR = os.path.join(ROOT, "models", "severity_classifier")
EPOCHS, BATCH, LR, MAX_LEN, SEED = 3, 32, 5e-4, 64, 42


def keyword_baseline(text: str) -> int:
    t = text.lower()
    if any(k in t for k in ("tornado", "super cyclone", "catastrophic", "evacuat", "red alert", "life-threatening", "historic")):
        return 3
    if any(k in t for k in ("warning", "orange alert", "severe", "dangerous", "damaging", "torrential")):
        return 2
    if any(k in t for k in ("advisory", "gusty", "fog", "thunderstorms possible", "waterlogging", "cold wave", "moderate")):
        return 1
    return 0


@torch.no_grad()
def predict(model, tok, texts, bs=64):
    model.eval()
    preds = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        preds += model(**enc).logits.argmax(-1).tolist()
    return np.array(preds)


def report(name, y_true, y_pred):
    acc = (y_true == y_pred).mean()
    f1s = []
    for c in range(len(LABELS)):
        tp = ((y_pred == c) & (y_true == c)).sum()
        p = tp / max((y_pred == c).sum(), 1)
        r = tp / max((y_true == c).sum(), 1)
        f1s.append(2 * p * r / max(p + r, 1e-9))
    print(f"  {name:<22} accuracy={acc:.3f}  macro-F1={np.mean(f1s):.3f}")
    return float(acc), float(np.mean(f1s))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train, test = build_dataset(seed=SEED)
    x_tr, y_tr = [t for t, _ in train], np.array([l for _, l in train])
    x_te, y_te = [t for t, _ in test], np.array([l for _, l in test])
    print(f"train={len(train)} test={len(test)} (test uses held-out hazard wording)")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    cfg = dict(num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
               label2id={l: i for i, l in enumerate(LABELS)})
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, **cfg)

    print("\nBaselines:")
    results = {"untrained": report("untrained DistilBERT", y_te, predict(base, tok, x_te))}
    results["keyword"] = report("keyword heuristic", y_te, np.array([keyword_baseline(t) for t in x_te]))

    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
                      target_modules=["q_lin", "v_lin"])
    model = get_peft_model(base, lora)  # classifier head is trained automatically for SEQ_CLS
    trainable, total = (sum(p.numel() for p in model.parameters() if p.requires_grad),
                        sum(p.numel() for p in model.parameters()))
    print(f"\nLoRA trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    steps = EPOCHS * ((len(x_tr) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(x_tr))
        losses = []
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            enc = tok([x_tr[j] for j in idx], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            out = model(**enc, labels=torch.tensor(y_tr[idx]))
            out.loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(out.loss.item())
        print(f"  epoch {epoch + 1}/{EPOCHS}  loss={np.mean(losses):.4f}  ({time.time() - start:.0f}s)")

    print("\nTest results:")
    y_pred = predict(model, tok, x_te)
    results["lora"] = report("LoRA fine-tuned", y_te, y_pred)

    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_te, y_pred):
        cm[t, p] += 1
    print("\nConfusion matrix (rows=true, cols=pred):", LABELS)
    print(cm)

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)  # adapter + classification head
    tok.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "training_meta.json"), "w") as f:
        json.dump({"base_model": BASE_MODEL, "labels": LABELS, "max_len": MAX_LEN,
                   "test_results": {k: {"accuracy": v[0], "macro_f1": v[1]} for k, v in results.items()},
                   "note": "Trained on synthetic data; not for operational use."}, f, indent=2)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
