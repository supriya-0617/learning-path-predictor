"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
from app.core import (
    COURSE_CATALOGUE,
    generate_mock_sequences,
    get_course_catalogue,
    milestone_1,
)

# Milestone 1: Generate mock student course completion data as sequences of course IDs.
def test_milestone_1():
    first = generate_mock_sequences(n_sequences=200, seed=123)
    second = generate_mock_sequences(n_sequences=200, seed=123)
    assert first == second
    assert len(first) == 200
    assert len(get_course_catalogue()) >= 10
    assert all(sequence and all(course in COURSE_CATALOGUE for course in sequence) for sequence in first)
    result = milestone_1(n_sequences=100, seed=123)
    assert result["count"] == 100
    assert len(result["held_out_sequences"]) == 20


def test_milestone_2(tmp_path):
    from app.core import milestone_2

    result = milestone_2(
        n_sequences=40,
        seed=123,
        epochs=1,
        directory=str(tmp_path / "model"),
    )
    assert (tmp_path / "model" / "course_sequence_model.pt").exists()
    assert (tmp_path / "model" / "course_vocabulary.json").exists()
    assert len(result["history"]) == 2
# Milestone 3: Implement a function to predict the next likely course given a partial sequence.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Create a Streamlit interface for students to input their completed courses and view recommendations.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Analyze which course "itemsets" the model has learned.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
