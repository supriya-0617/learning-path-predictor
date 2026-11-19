# Learning Path Predictor

Learning Path Predictor recommends a student's likely next course from the courses already completed. It combines generated student sequences, a genuine PyTorch causal language model, an `lfretail-lfmarketbasket`-style fine-tuning stage, held-out evaluation, and a Streamlit interface.

## Problem statement

Students often need a relevant course sequence but do not have a personalized prerequisite-aware plan. This application treats each student history as a course-ID sequence and uses learned next-token probabilities to rank unseen catalogue courses.

## How it works

1. Generate reproducible mock sequences from six realistic learning paths.
2. Pretrain a lightweight LSTM language model on the course-ID sequences.
3. Fine-tune with the `LFMarketBasketFineTuner`, which augments each basket-style sequence with course prefixes and co-occurring course sub-sequences and trains the same causal next-token objective.
4. Score every catalogue course for a supplied partial sequence.
5. Exclude completed courses and return the highest-ranked remaining course IDs.
6. Mine support-frequency pairs and sequential transitions from training data for evidence of learned relationships.

## Course catalogue

The 12-course catalogue includes:

- CS101 Programming Fundamentals
- PY101 Python Programming
- DSA201 Data Structures
- DB201 Database Management
- WEB201 Web Development
- ST201 Statistics for Data Science
- ML301 Machine Learning
- DS301 Data Science
- CLD301 Cloud Computing
- AI401 Artificial Intelligence
- DL401 Deep Learning
- MLO401 MLOps

## Mock data

`app/core.py` provides seeded Python `random` generation. The current generated dataset contains 1,200 sequences and is saved as:

- `data/course_catalogue.json`
- `data/course_sequences.json`

The fixed seed is 42. An 80/20 train/held-out split prevents evaluation leakage.

## Language model

The model is a CPU-portable PyTorch causal LSTM:

- Course and special tokens embedded in 32 dimensions
- One LSTM with 64 hidden units
- Linear vocabulary output head
- Next-token cross-entropy training
- Padding, beginning-of-sequence, and end-of-sequence tokens

The saved artifacts are `model/course_sequence_model.pt`, `model/course_vocabulary.json`, and `model/training_history.json`.

## `lfretail-lfmarketbasket` fine-tuning

The package `thilimarketbasket` is included as the project's drill dependency. Because its native MLX implementation is not available on Windows/CPU CI, the project preserves the technique's intended basket relationship and causal next-token objectives in `LFMarketBasketFineTuner` in `app/core.py`. It adds prefix examples and randomized co-occurring sub-sequences, while the course sequence remains the training target.

## Training

```bash
python -c "from app.core import milestone_1, milestone_2, milestone_5; milestone_1(); milestone_2(n_sequences=1200, seed=42, epochs=6); milestone_5()"
```

The initial model was pretrained and fine-tuned for six epochs per stage. Training history is recorded in `model/training_history.json`.

## Inference

```python
from app.core import predict_next_courses, recommendation_details

predict_next_courses(["CS101", "PY101"], top_n=5)
recommendation_details(["CS101", "PY101"], top_n=5)
```

Empty inputs receive a data-derived popularity ranking. Invalid IDs are ignored, and completed courses are removed from the candidate set.

## Streamlit interface

```bash
streamlit run app/main.py
```

The interface shows the course catalogue, uses `st.multiselect` for completed courses, accepts a top-N setting, and renders recommendations as a table.

## Evaluation

Evaluation uses 240 held-out next-course cases with the final course hidden. Metrics and sample predictions are saved in `model/evaluation.json`.

| Method | Top-1 accuracy | Top-3 hit rate | Top-5 hit rate | MRR |
|---|---:|---:|---:|---:|
| Fine-tuned course LM | 0.6708 | 0.9792 | 1.0000 | 0.8122 |
| Popularity baseline | 0.0458 | 0.2792 | 0.6708 | 0.2908 |

The model substantially outperformed the global popularity baseline.

## Sample recommendations

| Completed courses | Actual next course | Model top 5 |
|---|---|---|
| PY101 | DB201 | DB201, WEB201, ST201, DSA201, ML301 |
| ST201, DS301, ML301 | MLO401 | MLO401, AI401, DL401, PY101, DSA201 |
| CS101, PY101, DSA201 | DB201 | DB201, ST201, WEB201, DS301, ML301 |
| PY101, DB201, DS301 | CLD301 | CLD301, MLO401, DSA201, CS101, ML301 |

## Learned itemsets and relationships

The support-based analysis is saved in `model/itemset_analysis.json`.

Examples with at least 5% support in the 960-sequence training split include:

- CS101 + PY101: 28.96%
- ML301 + ST201: 27.40%
- DB201 + PY101: 26.77%
- PY101 + ST201: 24.27%
- DSA201 + PY101: 22.08%

The strongest sequential transitions include CS101 -> PY101 (278 examples), PY101 -> DSA201 (212), DS301 -> ML301 (173), PY101 -> DB201 (154), and ST201 -> DS301 (148).

These results are direct counts and model outputs. They show prerequisite-like relationships in the mock data, not evidence about real student populations.

## Project structure

```text
app/
  core.py
  llm.py
  main.py
  storage.py
data/
  course_catalogue.json
  course_sequences.json
model/
  course_sequence_model.pt
  course_vocabulary.json
  training_history.json
  evaluation.json
  itemset_analysis.json
tests/
```

## Installation

Requires Python 3.12 and Git.

```bash
python3.12 -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Optional LLM settings can be copied from `.env.example`. They are unrelated to course-model inference.

## Tests

```bash
pytest -q
```

Tests use lightweight fixtures and do not require full model retraining. The full saved model is available for local Streamlit execution and inference.

## Limitations

- Mock sequences cannot represent real student behavior.
- The held-out split is generated from the same seeded distributions.
- The LSTM does not provide explanations for every ranking.
- Rare courses and combinations may have limited training evidence.
- The model uses course-ID sequence order and does not currently encode instructor, duration, or student-specific information.
