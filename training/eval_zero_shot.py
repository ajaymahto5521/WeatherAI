"""LLM baselines for the severity classifier, scored on the same realistic test set.

Two prompts, no training examples in either:
  simple - just the four labels (the plain zero-shot comparison)
  rubric - the labels plus this project's SOP thresholds, an upper reference showing what
           an LLM can do when it is told the cut-offs the fine-tuned model learns from data

Results go to models/severity_classifier/training_meta.json as "zero_shot:<model>" and
"rubric:<model>", next to the untrained, keyword and LoRA results.

Run from the project root: python training/eval_zero_shot.py [model-name ...]
(default: the OPENAI_MODEL_NAME from .env)
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from app.llm_client import get_client
from training.severity_data_realistic import LABELS, build_dataset
from training.train_severity import OUT_DIR, SEED, report

SIMPLE = (
    "Classify the severity of this site weather alert as exactly one of: "
    + ", ".join(LABELS) + ". Reply with the label only."
)
RUBRIC = SIMPLE + (
    " Severity is set by the measured values, not the wording; use the worst hazard present. "
    "Wind (km/h; 1 m/s = 3.6 km/h): under 31 LOW, 31-44 MODERATE, 45-74 SEVERE, 75+ CRITICAL. "
    "Air temperature: under 32C LOW, 32-35 MODERATE, 36-39 SEVERE, 40+ CRITICAL. "
    "Heat index: under 32C LOW, 32-37 MODERATE, 38-42 SEVERE, 43+ CRITICAL. "
    "Standing water: under 5 cm LOW, 5-15 cm MODERATE, 16-30 cm SEVERE, over 30 cm CRITICAL "
    "(10 mm = 1 cm). Ignore humidity, visibility, pressure, cloud cover and times."
)
PROMPTS = {"zero_shot": SIMPLE, "rubric": RUBRIC}


def main():
    llm = get_client()
    if llm is None:
        sys.exit("OPENAI_API_KEY is not configured (see .env)")
    client, default_model = llm
    for model in sys.argv[1:] or [default_model]:
        for name, prompt in PROMPTS.items():
            evaluate(client, model, name, prompt)


def evaluate(client, model: str, name: str, prompt: str):
    _, test = build_dataset(seed=SEED)
    texts = [t for t, _ in test]
    y_true = np.array([l for _, l in test])

    def classify(text: str) -> int:
        reply = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        ).choices[0].message.content or ""
        reply = reply.strip().upper()
        # an unparseable reply counts as wrong, never as a lucky guess
        return LABELS.index(reply) if reply in LABELS else -1

    with ThreadPoolExecutor(max_workers=8) as pool:
        y_pred = np.array(list(pool.map(classify, texts)))

    print(f"{name} {model} on {len(texts)} test examples "
          f"({int((y_pred < 0).sum())} unparseable replies)")
    acc, f1 = report(f"{name} {model}", y_true, y_pred)

    meta_path = os.path.join(OUT_DIR, "training_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["test_results"][f"{name}:{model}"] = {"accuracy": acc, "macro_f1": f1}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated {meta_path}")


if __name__ == "__main__":
    main()
