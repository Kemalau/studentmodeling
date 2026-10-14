import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_summarizer_preserves_claim_guard(tmp_path: Path) -> None:
    t2 = {"metrics": {"overall": {"math_equivalence": 0.5}}}
    t4 = {"metrics": {"overall": {"joint_math_equivalence": 0.7}}}
    traditional = {
        "T2": {"metrics": {"overall": {"math_equivalence": 0.4}}},
        "T5": {"metrics": {"overall": {"joint_math_equivalence": 0.6}}},
    }
    matrix = {
        "rows": [
            {"model": "Qwen3-32B Base", "T2": {"value": 0.2}, "T4": {"value": 0.3}},
            {"model": "Unified Qwen3-32B QLoRA", "T2": {"value": 0.6}, "T4": {"value": 0.8}},
        ]
    }
    paths = {}
    for name, value in (("t2", t2), ("t4", t4), ("traditional", traditional), ("matrix", matrix)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            "python", str(ROOT / "scripts/summarize_neural_response_specialists.py"),
            "--t2", str(paths["t2"]), "--t4", str(paths["t4"]),
            "--traditional", str(paths["traditional"]), "--matrix", str(paths["matrix"]),
            "--output-json", str(output), "--output-md", str(tmp_path / "comparison.md"),
        ],
        check=True,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["status"] == "complete"
    assert value["cells"]["T2"]["unified_minus_neural"] == pytest.approx(0.1)
    assert value["cells"]["T2"]["significance_pending"] is True
    assert "Do not call" in value["claim_guard"]
