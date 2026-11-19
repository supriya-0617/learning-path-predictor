"""Learning Path Predictor milestones and shared model utilities."""

import json
import random
from pathlib import Path


COURSE_CATALOGUE = {
    "CS101": {"name": "Programming Fundamentals", "category": "Foundations"},
    "PY101": {"name": "Python Programming", "category": "Programming"},
    "DSA201": {"name": "Data Structures", "category": "Computer Science"},
    "DB201": {"name": "Database Management", "category": "Data"},
    "WEB201": {"name": "Web Development", "category": "Development"},
    "ST201": {"name": "Statistics for Data Science", "category": "Data"},
    "ML301": {"name": "Machine Learning", "category": "Machine Learning"},
    "DS301": {"name": "Data Science", "category": "Data"},
    "CLD301": {"name": "Cloud Computing", "category": "Cloud"},
    "AI401": {"name": "Artificial Intelligence", "category": "AI"},
    "DL401": {"name": "Deep Learning", "category": "AI"},
    "MLO401": {"name": "MLOps", "category": "Machine Learning"},
}

COURSE_IDS = tuple(COURSE_CATALOGUE)


def get_course_catalogue() -> dict:
    return {course_id: dict(details) for course_id, details in COURSE_CATALOGUE.items()}


def generate_mock_sequences(n_sequences: int = 1200, seed: int = 42) -> list[list[str]]:
    """Generate realistic, reproducible student learning paths with prerequisites."""
    if n_sequences < 1:
        return []

    rng = random.Random(seed)
    paths = (
        ["CS101", "PY101", "DSA201", "DB201", "WEB201"],
        ["CS101", "PY101", "DSA201", "ST201", "DS301", "ML301"],
        ["CS101", "PY101", "ST201", "ML301", "AI401", "DL401"],
        ["CS101", "PY101", "DB201", "DS301", "CLD301", "MLO401"],
        ["PY101", "WEB201", "DB201", "CLD301"],
        ["ST201", "DS301", "ML301", "MLO401"],
    )

    sequences = []
    for _ in range(n_sequences):
        path = list(rng.choice(paths))
        if rng.random() < 0.65:
            skip = rng.randrange(1, max(2, len(path) - 1))
            path = [course for i, course in enumerate(path) if i != skip or rng.random() < 0.25]

        start = rng.randrange(0, min(2, len(path)))
        sequence = path[start:]
        if len(sequence) > 1 and rng.random() < 0.35:
            sequence = sequence[:rng.randint(2, len(sequence))]

        if len(sequence) < 2:
            sequence = list(dict.fromkeys([rng.choice(COURSE_IDS), rng.choice(COURSE_IDS)]))
        if len(sequence) < 2:
            sequence = ["CS101", "PY101"]

        sequences.append(sequence)
    return sequences


def split_sequences(
    sequences: list[list[str]], train_ratio: float = 0.8, seed: int = 42
) -> tuple[list[list[str]], list[list[str]]]:
    rng = random.Random(seed)
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:split], shuffled[split:]


def save_milestone_1(
    directory: str = "data", n_sequences: int = 1200, seed: int = 42
) -> tuple[list[list[str]], dict]:
    data_dir = Path(directory)
    data_dir.mkdir(parents=True, exist_ok=True)
    catalogue = get_course_catalogue()
    sequences = generate_mock_sequences(n_sequences=n_sequences, seed=seed)
    (data_dir / "course_catalogue.json").write_text(
        json.dumps(catalogue, indent=2), encoding="utf-8"
    )
    (data_dir / "course_sequences.json").write_text(
        json.dumps({"seed": seed, "sequences": sequences}, indent=2), encoding="utf-8"
    )
    return sequences, catalogue


generate_course_catalogue = get_course_catalogue


# --- Milestone 1: Generate mock student course completion data as sequences of course IDs.
def milestone_1(n_sequences: int = 1200, seed: int = 42):
    sequences, catalogue = save_milestone_1(n_sequences=n_sequences, seed=seed)
    train, held_out = split_sequences(sequences, seed=seed)
    return {
        "sequences": sequences,
        "train_sequences": train,
        "held_out_sequences": held_out,
        "catalogue": catalogue,
        "seed": seed,
        "count": len(sequences),
    }
