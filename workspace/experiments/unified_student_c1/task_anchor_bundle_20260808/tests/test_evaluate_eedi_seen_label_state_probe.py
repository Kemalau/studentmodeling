import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_eedi_seen_label_state_probe import select_panel  # noqa: E402


def row(example_id: str, label: str, *, source: str = "eedi_mining") -> dict:
    return {
        "example_id": example_id,
        "task": "diagnose_misconception",
        "source": source,
        "target_label": label,
    }


def test_panel_is_fit_on_train_and_requires_formal_support() -> None:
    train = [row(f"a-{i}", "a") for i in range(5)] + [
        row(f"b-{i}", "b") for i in range(5)
    ] + [row(f"c-{i}", "c") for i in range(10)]
    evaluation = [row("a-test", "a"), row("b-test", "b")]
    selected_train, selected_eval, labels = select_panel(
        train, evaluation, min_train_per_label=5, max_train_per_label=5, seed=42
    )
    assert labels == ["a", "b"]
    assert len(selected_train) == 10
    assert {item["target_label"] for item in selected_eval} == {"a", "b"}


def test_panel_rejects_one_class() -> None:
    train = [row(f"a-{i}", "a") for i in range(5)]
    try:
        select_panel(train, [row("a-test", "a")], min_train_per_label=5, max_train_per_label=5, seed=42)
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("one-class probe panel was accepted")
