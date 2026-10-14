import json
from pathlib import Path

from scripts.materialize_state8_curriculum_cp4000_proactive_plan import ARMS, build


def write_receipt(root: Path, arm: str, marker: str) -> None:
    path = root / ARMS[arm] / "milestones/checkpoint-4000.completion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "step": 4000,
                "sha256": {
                    "adapter_model": marker * 64,
                    "recurrent_state": chr(ord(marker) + 1) * 64,
                },
            }
        )
    )


def test_builds_three_strict_state_only_parallel_branches(tmp_path: Path) -> None:
    write_receipt(tmp_path, "generic", "a")
    write_receipt(tmp_path, "current_query", "c")
    result = build(tmp_path)
    assert result["parallel_gpu_jobs"] == 3
    assert result["seed_base"] == 5201
    assert result["raw_history_visible_to_decoder"] is False
    assert result["random_state_target"] == "insufficient_evidence"
    assert result["scientific_floors_unchanged"] is True
    branches = {row["name"]: row for row in result["branches"]}
    assert set(branches) == {
        "semantic_binding",
        "specificity_triplet",
        "balanced_continuation",
    }
    assert branches["semantic_binding"]["source"]["arm"] == "current_query"
    assert branches["specificity_triplet"]["training_arm"] == "full"
    assert branches["balanced_continuation"]["source"]["arm"] == "generic"


def test_rejects_incomplete_checkpoint_receipt(tmp_path: Path) -> None:
    write_receipt(tmp_path, "generic", "a")
    write_receipt(tmp_path, "current_query", "c")
    path = (
        tmp_path
        / ARMS["current_query"]
        / "milestones/checkpoint-4000.completion.json"
    )
    value = json.loads(path.read_text())
    value["status"] = "partial"
    path.write_text(json.dumps(value))
    try:
        build(tmp_path)
    except ValueError as error:
        assert "Incomplete checkpoint-4000" in str(error)
    else:
        raise AssertionError("Incomplete proactive source was accepted")


def test_builds_disjoint_checkpoint3000_proactive_plan(tmp_path: Path) -> None:
    for arm, marker in (("generic", "a"), ("current_query", "c")):
        path = root = tmp_path / ARMS[arm] / "milestones/checkpoint-3000.completion.json"
        root.parent.mkdir(parents=True, exist_ok=True)
        root.write_text(json.dumps({
            "status": "complete",
            "step": 3000,
            "sha256": {
                "adapter_model": marker * 64,
                "recurrent_state": chr(ord(marker) + 1) * 64,
            },
        }))
    result = build(tmp_path, checkpoint_step=3000, seed_base=6201)
    assert result["proactive_checkpoint_step"] == 3000
    assert result["proactive_before_state_only_boundary"] is True
    assert result["proactive_from_state_only_boundary"] is False
    assert result["seed_base"] == 6201
    assert {
        branch["source"]["checkpoint_step"] for branch in result["branches"]
    } == {3000}
