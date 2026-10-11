"""Harder, noisier severity data whose labels follow this project's site SOP thresholds.

The label is computed from the measured values (wind, heat, flood water), using the
cut-offs in data/reference/{severe_wind_and_storm_sop,heatwave_safety_protocol,
flood_emergency_sop}.md. The wording is deliberately messy:

  - mixed units (km/h and m/s, mm and cm), values packed near the cut-offs
  - distractor numbers (humidity, visibility, pressure) that do not affect the label
  - hype words that match the true level only half the time
  - one or two hazards per alert; the label is the worst one
  - typos, lower-casing, abbreviations
  - test frames, hype and context phrases never seen in training

A zero-shot LLM does not know a site's cut-offs, so it has to guess from the wording.
A fine-tuned model can learn them from the labels. Still generated text: it says nothing
about real-world accuracy.
"""

from __future__ import annotations

import random

LABELS = ["LOW", "MODERATE", "SEVERE", "CRITICAL"]

# SOP scale as (lower bound, label id) in the unit shown; the highest bound reached wins.
WIND_KMH = [(0, 0), (31, 1), (45, 2), (75, 3)]       # Normal / Caution / High risk+Severe / Critical
HEAT_AMBIENT_C = [(-50, 0), (32, 1), (36, 2), (40, 3)]  # <32 / 32-35 / 36-39 / >=40
HEAT_INDEX_C = [(-50, 0), (32, 1), (38, 2), (43, 3)]    # <32 / 32-37 / 38-42 / >=43
FLOOD_CM = [(0, 0), (5, 1), (16, 2), (31, 3)]           # <5 / 5-15 / 16-30 / >30


def level(value: float, scale: list[tuple[float, int]]) -> int:
    lvl = 0
    for bound, lab in scale:
        if value >= bound:
            lvl = lab
    return lvl


# value ranges per level, used to draw a value that lands in the wanted level
RANGES = {
    "wind": [(3, 30), (31, 44), (45, 74), (75, 150)],
    "heat": [(14, 31), (32, 35), (36, 39), (40, 49)],
    "hindex": [(20, 31), (32, 37), (38, 42), (43, 56)],
    "flood": [(0, 4), (5, 15), (16, 30), (31, 90)],
}
SCALES = {"wind": WIND_KMH, "heat": HEAT_AMBIENT_C, "hindex": HEAT_INDEX_C, "flood": FLOOD_CM}


def draw(kind: str, want: int, rng: random.Random) -> int:
    """A value inside level `want`; half the time within 3 of a cut-off."""
    lo, hi = RANGES[kind][want]
    if rng.random() < 0.5:
        edge = rng.choice([lo, hi])
        return max(lo, min(hi, edge + rng.randint(-3, 3)))
    return rng.randint(lo, hi)


# (frame, kind) text per measurement; {v} is the number shown, in the unit named.
MEASURE_TRAIN = {
    "wind": ["winds {v} km/h", "gusts up to {v} km/h", "sustained wind {v} km/h", "wind speed {v} km/h",
             "winds of {ms} m/s", "gusting {ms} m/s"],
    "heat": ["temperature {v}°C", "max temp {v}°C", "air temp reaching {v}°C", "mercury at {v}°C"],
    "hindex": ["heat index {v}°C", "feels-like {v}°C", "heat index near {v}°C"],
    "flood": ["standing water {v} cm", "water depth {v} cm", "{mm} mm of water on site roads",
              "water level up {v} cm"],
}
MEASURE_TEST = {
    "wind": ["wind {v} km/h", "anemometer reads {ms} m/s", "squalls to {v} km/h"],
    "heat": ["{v}°C ambient", "dry-bulb {v}°C", "afternoon high {v}°C"],
    "hindex": ["HI {v}°C", "apparent temperature {v}°C"],
    "flood": ["ponding {v} cm deep", "{mm} mm water at the gate", "depth gauge {v} cm"],
}

# hype wording per level; train and test sets are disjoint
HYPE_TRAIN = [
    ["Fair conditions", "Quiet shift", "Routine weather", "Nothing unusual"],
    ["Unsettled weather", "Conditions deteriorating", "Be alert", "Worsening conditions"],
    ["Dangerous conditions", "Serious weather alert", "High risk", "Alert issued"],
    ["Extreme emergency", "Life-threatening", "Red alert", "Catastrophic risk"],
]
HYPE_TEST = [
    ["Calm outlook", "Smooth operations", "Benign day"],
    ["Changeable skies", "Keep watch", "Turning rough"],
    ["Hazardous setting", "Warning in force", "Threatening skies"],
    ["Maximum danger", "Critical situation", "Disaster-level event"],
]

CONTEXT_TRAIN = ["Site A", "Tower 2 yard", "Plant gate", "North block", "Depot", "Delhi NCR sites"]
CONTEXT_TEST = ["East camp", "Riverside works", "Warehouse 7", "Phase 3 crew"]

DISTRACTORS = [
    "humidity {h}%", "visibility {vis} km", "pressure {p} hPa", "update at {t}", "cloud cover {c}%",
]
ADVICE_TRAIN = ["Stay informed.", "Supervisors to brief crews.", "Check again at the next update.", ""]
ADVICE_TEST = ["Report to the control room.", "Review the plan.", "Log the reading.", ""]


def _typo(text: str, rng: random.Random) -> str:
    """Swap two adjacent letters inside a few words; digits are never touched."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 4 and w.isalpha() and rng.random() < 0.08:
            j = rng.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _phrase(kind: str, value: int, frames: list[str], rng: random.Random) -> tuple[str, int]:
    """Text for one measurement and the level the SHOWN number implies."""
    frame = rng.choice(frames)
    if "{ms}" in frame:
        ms = round(value / 3.6, 1)
        return frame.format(ms=ms), level(round(ms * 3.6), WIND_KMH)
    if "{mm}" in frame:
        mm = value * 10
        return frame.format(mm=mm), level(mm / 10, FLOOD_CM)
    return frame.format(v=value), level(value, SCALES[kind])


def make_example(want: int, rng: random.Random, test: bool) -> tuple[str, int]:
    measures, hype, ctx, advice = (
        (MEASURE_TEST, HYPE_TEST, CONTEXT_TEST, ADVICE_TEST) if test
        else (MEASURE_TRAIN, HYPE_TRAIN, CONTEXT_TRAIN, ADVICE_TRAIN)
    )
    kinds = rng.sample(list(RANGES), rng.choice([1, 1, 2]))
    levels, parts = [], []
    for n, kind in enumerate(kinds):
        lvl = want if n == 0 else rng.randint(0, want)  # extra hazards never exceed the target
        text, shown = _phrase(kind, draw(kind, lvl, rng), measures[kind], rng)
        levels.append(shown)
        parts.append(text)
    label = max(levels)

    # hype matches the real level only half the time
    hype_level = label if rng.random() < 0.5 else rng.randrange(4)
    sentences = [f"{rng.choice(ctx)}: {rng.choice(hype[hype_level])}", ", ".join(parts)]
    for _ in range(rng.choice([0, 1, 2])):
        sentences.append(rng.choice(DISTRACTORS).format(
            h=rng.randint(20, 98), vis=rng.randint(1, 12), p=rng.randint(985, 1020),
            t=f"{rng.randint(0, 23):02d}:{rng.choice(['00', '15', '30', '45'])}", c=rng.randint(0, 100)))
    adv = rng.choice(advice)
    text = ". ".join(sentences) + "." + (f" {adv}" if adv else "")
    if rng.random() < 0.15:
        text = text.lower()
    return _typo(text, rng), label


def build_dataset(n_per_class_train: int = 1000, n_per_class_test: int = 100, seed: int = 42):
    """Returns (train, test) lists of (text, label_id), balanced by label."""
    rng = random.Random(seed)
    train, test = [], []
    for want in range(4):
        # draw until the SHOWN values give exactly the wanted label, so labels stay exact
        for bucket, n, is_test in ((train, n_per_class_train, False), (test, n_per_class_test, True)):
            while sum(1 for _, l in bucket if l == want) < n:
                text, label = make_example(want, rng, is_test)
                if label == want:
                    bucket.append((text, label))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


if __name__ == "__main__":
    tr, te = build_dataset(5, 3)
    for t, l in te:
        print(LABELS[l].ljust(9), t)
