from __future__ import annotations

import json
from pathlib import Path

import train_state8_diagnosis_first_eedi_curriculum as curriculum
from train_recurrent_student_m4 import build_repeated_causal_schedule


def test_causal_schedule_is_confined_to_final_state_only_quintile() -> None:
    schedule = curriculum.shifted_repeated_causal_schedule(
        build_repeated_causal_schedule,
        real_count=5000,
        causal_count=450,
        fraction=0.10,
        seed=83,
    )
    assert len(schedule) == 450
    assert min(schedule) >= 4000
    assert max(schedule) < 5000
    assert set(schedule.values()) == set(range(450))


def test_eedi_merge_namespaces_splits_and_preserves_controls(
    tmp_path: Path, monkeypatch
) -> None:
    real_train = tmp_path / "train.jsonl"
    real_valid = tmp_path / "valid.jsonl"
    causal_train = tmp_path / "causal_train.jsonl"
    causal_valid = tmp_path / "causal_valid.jsonl"
    for path in (real_train, real_valid, causal_train, causal_valid):
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        curriculum,
        "cli_path",
        lambda flag: {
            "--train-jsonl": real_train,
            "--valid-jsonl": real_valid,
            "--causal-train-jsonl": causal_train,
            "--causal-valid-jsonl": causal_valid,
        }[flag],
    )
    replaced: dict[str, Path] = {}
    monkeypatch.setattr(
        curriculum, "replace_cli_path", lambda flag, path: replaced.update({flag: path})
    )
    monkeypatch.setattr(
        curriculum,
        "read_jsonl",
        lambda path: (
            [{"pair_id": "base-train", "source": "malrule"}]
            if path == causal_train
            else [{"pair_id": "base-valid", "source": "malrule"}]
            if path == causal_valid
            else [{"example_id": str(path)}]
        ),
    )

    def fake_twins(rows, *, twins, seed, offset):
        del rows, seed
        output = []
        for index in range(twins * 2):
            output.append(
                {
                    "pair_id": f"builder:{index}",
                    "source": "eedi_mining",
                    "same_question_query_identical": True,
                    "raw_history_visible_to_decoder": False,
                }
            )
        return output, {"twins": twins, "offset": offset}

    monkeypatch.setattr(curriculum, "build_twins", fake_twins)
    output_dir = tmp_path / "output"
    receipt = curriculum.materialize_eedi_causal_data(output_dir)
    train_rows = curriculum.read_jsonl(replaced["--causal-train-jsonl"])
    valid_rows = curriculum.read_jsonl(replaced["--causal-valid-jsonl"])
    # Restore the real reader because it was mocked for source materialization.
    with replaced["--causal-train-jsonl"].open(encoding="utf-8") as handle:
        train_rows = [json.loads(line) for line in handle if line.strip()]
    with replaced["--causal-valid-jsonl"].open(encoding="utf-8") as handle:
        valid_rows = [json.loads(line) for line in handle if line.strip()]
    train_ids = {row["pair_id"] for row in train_rows}
    valid_ids = {row["pair_id"] for row in valid_rows}
    assert train_ids.isdisjoint(valid_ids)
    assert receipt["eedi_train_pairs"] == 200
    assert receipt["eedi_valid_pairs"] == 50
    assert receipt["same_question_query_identical"] is True
    assert receipt["raw_history_visible_to_decoder"] is False
