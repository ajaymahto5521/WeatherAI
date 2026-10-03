"""
app/models/severity.py
======================
Inference wrapper for the LoRA-fine-tuned severity classifier (Milestone 5).

Loads the DistilBERT base model, attaches the saved LoRA adapter, and classifies
weather text as LOW / MODERATE / SEVERE / CRITICAL with a confidence score.

Trained on synthetic data — educational use only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SeverityResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


class SeverityClassifier:
    def __init__(self, model_dir: str | Path | None = None):
        if model_dir is None:
            from app.config import settings
            model_dir = settings.severity_model_path
        self._dir = Path(model_dir)
        self._model = None
        self._tok = None
        self._labels: list[str] = []
        self._max_len = 64

    def _load(self) -> None:
        meta_path = self._dir / "training_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Severity classifier not found at {self._dir}. "
                "Run: python training/train_severity.py"
            )
        # heavy imports kept local so the API starts fast without this model
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads(meta_path.read_text("utf-8"))
        self._labels, self._max_len = meta["labels"], meta["max_len"]
        base = AutoModelForSequenceClassification.from_pretrained(
            meta["base_model"],
            num_labels=len(self._labels),
            id2label=dict(enumerate(self._labels)),
            label2id={l: i for i, l in enumerate(self._labels)},
        )
        self._model = PeftModel.from_pretrained(base, str(self._dir)).eval()
        self._tok = AutoTokenizer.from_pretrained(str(self._dir))
        logger.info("Severity classifier loaded from %s", self._dir)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def classify(self, text: str) -> SeverityResult:
        import torch

        if not self.is_loaded:
            self._load()
        enc = self._tok(text, truncation=True, max_length=self._max_len, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self._model(**enc).logits[0], dim=-1).tolist()
        best = max(range(len(probs)), key=probs.__getitem__)
        return SeverityResult(
            label=self._labels[best],
            confidence=round(probs[best], 4),
            probabilities={l: round(p, 4) for l, p in zip(self._labels, probs)},
        )
