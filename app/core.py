"""Learning Path Predictor milestones and shared model utilities."""

import json
import random
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


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


if nn is not None:
    class CourseSequenceLM(nn.Module):
        """Small causal LSTM language model for course-ID sequences."""

        def __init__(self, vocab_size: int, embedding_size: int = 32, hidden_size: int = 64):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, token_ids):
            embedded = self.embedding(token_ids)
            hidden, _ = self.lstm(embedded)
            return self.output(hidden)


class LFMarketBasketFineTuner:
    """LF market-basket adaptation: causal course-prefix training plus co-occurrence examples.

    In each fine-tuning step, every prefix predicts the next course, and random itemset-style
    sub-sequences reinforce related courses that occur together. It is the same next-token
    relationship learning used by the lfretail-lfmarketbasket approach, adapted to CPU/PyTorch.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def batches(self, sequences: list[list[str]], course_ids: list[str], augment: bool):
        rng = random.Random(self.seed)
        examples = []
        for sequence in sequences:
            sequence = list(dict.fromkeys(sequence))
            if augment:
                for prefix_length in range(1, len(sequence)):
                    examples.append(sequence[:prefix_length] + [sequence[prefix_length]])
                if len(sequence) >= 3:
                    size = rng.randint(2, len(sequence) - 1)
                    examples.append(rng.sample(sequence, size) + [rng.choice(sequence[size:])])
            else:
                examples.append(sequence)
        return examples


def _model_paths(directory: str = "model") -> tuple[Path, Path, Path]:
    model_dir = Path(directory)
    return model_dir, model_dir / "course_sequence_model.pt", model_dir / "course_vocabulary.json"


def load_course_model(directory: str = "model"):
    if torch is None:
        raise RuntimeError("PyTorch is required to train or load the course sequence model.")
    _, model_path, vocabulary_path = _model_paths(directory)
    if not model_path.exists() or not vocabulary_path.exists():
        raise FileNotFoundError("Fine-tuned course model files are unavailable.")
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    model = CourseSequenceLM(len(vocabulary["token_to_id"]))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, vocabulary


def _train_course_model(
    sequences: list[list[str]],
    epochs: int,
    seed: int,
    fine_tune: bool,
) -> tuple[object, dict, list[dict]]:
    if torch is None:
        raise RuntimeError("PyTorch is required to train the course sequence model.")

    torch.manual_seed(seed)
    random.seed(seed)
    token_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
    for course_id in COURSE_IDS:
        token_to_id[course_id] = len(token_to_id)
    id_to_token = {str(index): token for token, index in token_to_id.items()}

    model = CourseSequenceLM(len(token_to_id))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    history = []

    stages = [("pretrain", False)]
    if fine_tune:
        stages.append(("lfretail-lfmarketbasket", True))

    for stage_name, augment in stages:
        for epoch in range(epochs):
            rng = random.Random(seed + epoch)
            rng.shuffle(sequences)
            examples = LFMarketBasketFineTuner(seed + epoch).batches(
                sequences, list(COURSE_IDS), augment=augment
            )
            total_loss = 0.0
            count = 0
            for example in examples:
                input_ids = [1] + [token_to_id[course] for course in example]
                target_ids = input_ids[1:] + [2]
                inputs = torch.tensor([input_ids], dtype=torch.long)
                targets = torch.tensor([target_ids], dtype=torch.long)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = loss_function(logits[:, :-1, :].reshape(-1, len(token_to_id)),
                                     targets[:, :-1].reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                count += 1
            history.append({
                "stage": stage_name,
                "epoch": epoch + 1,
                "loss": total_loss / max(1, count),
            })
    return model, {"token_to_id": token_to_id, "id_to_token": id_to_token}, history


def save_course_model(model, vocabulary: dict, history: list[dict], directory: str = "model") -> dict:
    model_dir = Path(directory)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "course_sequence_model.pt"
    vocabulary_path = model_dir / "course_vocabulary.json"
    torch.save(model.state_dict(), model_path)
    vocabulary_path.write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    (model_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"model": str(model_path), "vocabulary": str(vocabulary_path), "history": history}


# --- Milestone 2: Pretrain a small language model and apply lfretail-lfmarketbasket.
def milestone_2(
    n_sequences: int = 1200,
    seed: int = 42,
    epochs: int = 12,
    directory: str = "model",
):
    sequences = generate_mock_sequences(n_sequences=n_sequences, seed=seed)
    train, _ = split_sequences(sequences, seed=seed)
    model, vocabulary, history = _train_course_model(
        train, epochs=epochs, seed=seed, fine_tune=True
    )
    return save_course_model(model, vocabulary, history, directory)


def course_name(course_id: str) -> str:
    details = COURSE_CATALOGUE.get(course_id)
    return details["name"] if details else str(course_id)


def predict_next_courses(
    completed_courses: list[str] | tuple[str, ...] | None,
    top_n: int = 5,
    directory: str = "model",
) -> list[str]:
    """Use the fine-tuned causal LM to rank valid, not-yet-completed courses."""
    if torch is None:
        raise RuntimeError("PyTorch is required for course recommendations.")

    top_n = max(1, int(top_n))
    valid = [] if completed_courses is None else [
        str(course_id).strip().upper()
        for course_id in completed_courses
        if str(course_id).strip().upper() in COURSE_CATALOGUE
    ]
    completed = list(dict.fromkeys(valid))

    model, vocabulary = load_course_model(directory)
    token_to_id = vocabulary["token_to_id"]
    available = [course_id for course_id in COURSE_IDS if course_id not in completed]
    if not available:
        return []

    if not completed:
        sequences = generate_mock_sequences(n_sequences=1200, seed=42)
        counts = {course_id: 0 for course_id in available}
        for sequence in sequences:
            for course_id in set(sequence):
                if course_id in counts:
                    counts[course_id] += 1
        ranked = sorted(available, key=lambda course_id: (-counts[course_id], course_id))
        return ranked[:top_n]

    input_ids = [1] + [token_to_id[course_id] for course_id in completed]
    inputs = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(inputs)[0, -1]
        probabilities = torch.softmax(logits, dim=-1)

    scores = [
        (float(probabilities[token_to_id[course_id]].item()), course_id)
        for course_id in available
    ]
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [course_id for _, course_id in scores[:top_n]]


def recommendation_details(
    completed_courses: list[str] | tuple[str, ...] | None,
    top_n: int = 5,
    directory: str = "model",
) -> list[dict]:
    return [
        {
            "course_id": course_id,
            "course_name": course_name(course_id),
            "category": COURSE_CATALOGUE[course_id]["category"],
        }
        for course_id in predict_next_courses(completed_courses, top_n, directory)
    ]


# --- Milestone 3: Implement next-course prediction.
def milestone_3(
    partial_sequence: list[str] | tuple[str, ...] | None,
    top_n: int = 5,
    directory: str = "model",
):
    recommendation_ids = predict_next_courses(partial_sequence, top_n, directory)
    return {
        "recommendations": recommendation_ids,
        "details": [
            {
                "course_id": course_id,
                "course_name": course_name(course_id),
                "category": COURSE_CATALOGUE[course_id]["category"],
            }
            for course_id in recommendation_ids
        ],
    }
