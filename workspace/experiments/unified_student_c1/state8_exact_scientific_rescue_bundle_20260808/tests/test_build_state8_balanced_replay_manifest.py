from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_state8_balanced_replay_manifest import QUOTAS  # noqa: E402


def test_balanced_replay_has_all_tasks_and_exact_support() -> None:
    tasks = {}
    for (task, _source, _label), count in QUOTAS.items():
        tasks[task] = tasks.get(task, 0) + count
    assert tasks == {
        "diagnose_misconception": 2500,
        "predict_correctness": 800,
        "predict_error_persistence": 500,
        "predict_prospective_error": 600,
        "predict_response": 600,
    }
    assert sum(tasks.values()) == 5000


def test_t6_is_globally_balanced_and_all_sources_are_covered() -> None:
    labels = {}
    sources = set()
    for (_task, source, label), count in QUOTAS.items():
        sources.add(source)
        if label is not None:
            labels[label] = labels.get(label, 0) + count
    assert labels == {"persistent": 250, "transient": 250}
    assert sources == {"dbe_kt22", "eedi_mining", "eedi_response", "foundationalassist", "malrule", "xes3g5m"}
