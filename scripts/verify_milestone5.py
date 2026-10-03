"""Verify the trained severity classifier. Run from project root."""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from app.models.severity import SeverityClassifier

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  [OK]   {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1


try:
    SeverityClassifier("models/does_not_exist").classify("x")
    check("missing model raises FileNotFoundError", False)
except FileNotFoundError:
    check("missing model raises FileNotFoundError", True)

meta = json.load(open("models/severity_classifier/training_meta.json"))
check("held-out accuracy recorded >= 0.80", meta["test_results"]["lora"]["accuracy"] >= 0.80,
      str(meta["test_results"]["lora"]))

clf = SeverityClassifier()
cases = [
    ("Clear skies and a gentle breeze. No precautions needed.", "LOW"),
    ("Evacuate immediately: catastrophic cyclone landfall with winds 180 km/h.", "CRITICAL"),
]
for text, expected in cases:
    r = clf.classify(text)
    check(f"'{text[:35]}...' -> {expected}", r.label == expected, str(r))
r = clf.classify(cases[0][0])
check("probabilities sum to ~1", abs(sum(r.probabilities.values()) - 1) < 0.01)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
