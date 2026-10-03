"""
training/severity_data.py
=========================
Synthetic dataset for the severity classifier (LOW / MODERATE / SEVERE / CRITICAL).

Each example is assembled from: hazard phrase + numeric measurement + impact + advice.
The numeric ranges and phrase pools overlap in vocabulary across levels (e.g. "rain"
appears at every level) so the model has to use intensity cues, not one keyword.

The test split holds out whole hazard phrases (and impact phrases) that never
appear in training, so test accuracy measures generalisation to unseen wording,
not memorisation of templates.

THIS IS SYNTHETIC. Scores on it say nothing about real-world performance.
"""

from __future__ import annotations

import random

LABELS = ["LOW", "MODERATE", "SEVERE", "CRITICAL"]

# level -> (hazard phrases, measurement templates, impacts, advice)
POOLS: dict[str, dict[str, list[str]]] = {
    "LOW": {
        "hazard": [
            "Sunny spells and light winds", "Cool and dry night ahead", "Slight chance of a brief shower", "Pleasant morning followed by scattered clouds",
            "Clear skies with a light breeze", "Partly cloudy and pleasant", "Light drizzle in the evening",
            "Calm and sunny conditions", "Mild haze in the morning", "A few passing clouds",
            "Gentle breeze and comfortable weather", "Fair weather across the region",
        ],
        "measure": ["Winds {w} km/h.", "Rainfall {r} mm.", "Temperature around {t}°C.", "Visibility above 8 km."],
        "impact": ["No disruption expected.", "Outdoor activities are fine.", "Roads and flights operate normally.", ""],
        "advice": ["No precautions needed.", "Enjoy the day.", ""],
        "ranges": {"w": (3, 22), "r": (0, 5), "t": (18, 32)},
    },
    "MODERATE": {
        "hazard": [
            "Heavy showers likely with poor visibility on highways", "Hot afternoon with an elevated heat index", "Squally winds expected near the coast", "Frost and chilly winds in the early hours",
            "Gusty winds and scattered thunderstorms", "Dense fog reducing visibility", "Hot and humid with a heat advisory",
            "Steady moderate rain through the day", "Thunderstorms possible this afternoon", "Strong winds with dust haze",
            "Cold wave conditions overnight", "Persistent showers with waterlogging in low areas",
        ],
        "measure": ["Winds {w} km/h.", "Rainfall {r} mm.", "Temperature reaching {t}°C.", "Visibility down to 300 m."],
        "impact": ["Minor travel delays likely.", "Some local waterlogging possible.", "Outdoor work may be uncomfortable.", "Small branches may fall."],
        "advice": ["Stay alert for updates.", "Drive carefully.", "Carry an umbrella and stay hydrated."],
        "ranges": {"w": (30, 52), "r": (20, 45), "t": (36, 40)},
    },
    "SEVERE": {
        "hazard": [
            "Red-flag lightning and large hail threat", "Tropical storm bringing destructive gusts", "Extremely heavy downpour with river levels rising fast", "Severe dust storm with near-zero visibility",
            "Severe thunderstorm warning with hail", "Flash flood warning for river basins", "Dangerous heatwave warning",
            "Damaging winds and torrential rain", "Orange alert for extremely heavy rainfall", "Severe cyclonic storm approaching the coast",
            "Violent squalls with lightning strikes", "Blizzard conditions with whiteout",
        ],
        "measure": ["Winds {w} km/h.", "Rainfall {r} mm in 24 hours.", "Temperature soaring to {t}°C.", "Gusts expected to exceed {w} km/h."],
        "impact": ["Trees and power lines likely to come down.", "Widespread flooding of roads and underpasses.", "Crop and property damage expected.", "Risk of heatstroke for all age groups."],
        "advice": ["Avoid travel and stay indoors.", "Move away from rivers and low-lying areas.", "Secure loose objects and avoid open fields."],
        "ranges": {"w": (65, 95), "r": (70, 120), "t": (42, 46)},
    },
    "CRITICAL": {
        "hazard": [
            "Emergency declared as a violent tornado touches down", "Category 5 hurricane approaching populated coastline", "Unprecedented deluge as embankments fail", "Deadly heat dome with temperatures beyond human tolerance",
            "Tornado emergency with confirmed tornado on the ground", "Extremely severe cyclone making landfall",
            "Catastrophic flooding with dam breach risk", "Red alert: life-threatening extreme heat",
            "Major hurricane storm surge threatens coast", "Devastating tornado outbreak underway",
            "Historic record-breaking flood event", "Super cyclone with destructive core approaching",
        ],
        "measure": ["Winds {w} km/h.", "Rainfall {r} mm in 24 hours.", "Temperature {t}°C.", "Storm surge above 4 metres."],
        "impact": ["Massive destruction of buildings and infrastructure expected.", "Imminent threat to life.",
                   "Entire communities may be cut off for days.", "Complete power and communication failure likely."],
        "advice": ["Evacuate immediately as ordered.", "Take shelter now in the strongest available structure.",
                   "Follow evacuation orders from authorities without delay."],
        "ranges": {"w": (125, 220), "r": (200, 400), "t": (47, 51)},
    },
}


def _make(level: str, hazard: str, impact_pool: list[str], rng: random.Random) -> str:
    p = POOLS[level]
    lo = lambda k: rng.randint(*p["ranges"][k])  # noqa: E731
    measure = rng.choice(p["measure"]).format(w=lo("w"), r=lo("r"), t=lo("t"))
    parts = [hazard + ".", measure, rng.choice(impact_pool), rng.choice(p["advice"])]
    # randomly drop optional parts so length/format varies
    parts = [parts[0]] + [x for x in parts[1:] if x and rng.random() > 0.2]
    return " ".join(parts)


def build_dataset(
    n_per_class_train: int = 300,
    n_per_class_test: int = 60,
    seed: int = 42,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Returns (train, test) lists of (text, label_id). Test wording is held out."""
    rng = random.Random(seed)
    train, test = [], []
    for label_id, level in enumerate(LABELS):
        hazards = POOLS[level]["hazard"][:]
        impacts = [i for i in POOLS[level]["impact"] if i]
        rng.shuffle(hazards)
        rng.shuffle(impacts)
        h_cut, i_cut = max(1, len(hazards) * 3 // 4), max(1, len(impacts) * 3 // 4)
        for _ in range(n_per_class_train):
            train.append((_make(level, rng.choice(hazards[:h_cut]), impacts[:i_cut], rng), label_id))
        for _ in range(n_per_class_test):
            test.append((_make(level, rng.choice(hazards[h_cut:]), impacts[i_cut:] or impacts, rng), label_id))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
