from pathlib import Path

import yaml

from scripts.validate_state8_taxonomy_teacher_curriculum_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_taxonomy_curriculum_amlt_is_fail_closed() -> None:
    validate(ROOT / "amlt/feeds_state8_taxonomy_teacher_curriculum_20260807.yaml")
    validate(ROOT / "amlt/feeds_state8_taxonomy_balanced_curriculum_20260807.yaml")


def test_taxonomy_curriculum_accepts_exact_amlt_archive_shape(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (ROOT / "amlt/feeds_state8_taxonomy_balanced_curriculum_20260807.yaml").read_text()
    )
    job = source["jobs"][0]
    job["target"] = {
        "cluster": "Feeds",
        "resource_group": "rg-cs-newsandfeeds-singularity",
        "service": "sing",
        "subscription_id": "b6dc87f3-c479-49c8-8cb5-7896da3ff895",
        "workspace_name": "CS-NewsAndFeeds-Singularity",
        "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
        "workspace_subscription_id": "b6dc87f3-c479-49c8-8cb5-7896da3ff895",
    }
    job["environment"] = source["environment"]
    job["environment"]["image"] = job["environment"]["image"].rsplit(":", 1)[0]
    job["storage"] = source["storage"]
    job["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = (
        "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895/"
        "resourceGroups/AMLStudio/providers/Microsoft.ManagedIdentity/"
        "userAssignedIdentities/rankfun_aml"
    )
    archived = tmp_path / "amulet.yaml"
    archived.write_text(yaml.safe_dump({"jobs": [job]}))
    validate(archived)


def test_taxonomy_curriculum_binds_same_rule_v2_rows() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert "e2a57273bd4c49229efc8cec522fc221e460174c388c43e2b0d4888cc5079225" in launcher
    assert "8bd0dc3ec367ce30e5e5da3526681c21f852ba991394cac423682cd721f48c8e" in launcher
    assert "a170b085858b656432725cd6e835703cb195c469d8374c3abfd9f0a51f909f24" not in launcher
    assert "d1f97a387b6be30467918813475edac6ac0d38e5923767f27f3aca792b130696" not in launcher


def test_taxonomy_curriculum_uses_preemption_resilient_checkpoint_cadence() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert 'STATE8_SAVE_STEPS="${STATE8_SAVE_STEPS:-125}"' in launcher
    assert "STATE8_SAVE_STEPS % 5 == 0" in launcher
    assert 'RESUME_CADENCE_ARGS=(--save-steps "$STATE8_SAVE_STEPS" --recovery-save-steps 5)' in launcher
    assert '--expected-planned-optimizer-steps "$TOTAL_ROUNDS"' in launcher
    for path in (
        ROOT / "amlt/feeds_state8_taxonomy_teacher_curriculum_20260807.yaml",
    ):
        assert "STATE8_SAVE_STEPS: '50'" in path.read_text()
    assert "STATE8_SAVE_STEPS: '5'" in (
        ROOT / "amlt/feeds_state8_taxonomy_balanced_curriculum_20260807.yaml"
    ).read_text()


def test_historical_balanced_archive_is_validated_against_its_old_cadence(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (ROOT / "amlt/feeds_state8_taxonomy_balanced_curriculum_20260807.yaml").read_text()
    )
    source["jobs"][0]["submit_args"]["env"]["STATE8_SAVE_STEPS"] = "50"
    path = tmp_path / "historical.yaml"
    path.write_text(yaml.safe_dump(source))
    validate(path, expected_save_steps="50")


def test_taxonomy_curriculum_uses_receipt_last_data_cache_with_fallback() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert "state8_curriculum_data_cache.py verify" in launcher
    assert "state8_taxonomy_curriculum_data_v1" in launcher
    assert "stage data_cache_verified" in launcher
    assert "elif [[ ! -f" in launcher
    assert '--train-jsonl "$SUBSET" --valid-jsonl "$VALID_JSONL"' in launcher
    assert 'ln -s "$DATA_CACHE_ROOT/common/valid.jsonl" "$VALID_JSONL"' in launcher
    assert 'ln -s "$DATA_CACHE_ROOT/$STATE8_REPLAY_VARIANT/train5000.jsonl" "$SUBSET"' in launcher


def test_taxonomy_curriculum_quarantines_only_proven_incompatible_receipts() -> None:
    launcher = (ROOT / "scripts/run_state8_qualified_teacher_curriculum.sh").read_text()
    assert "quarantine_incompatible_checkpoint_receipts.py" in launcher
    assert '--expected-planned-optimizer-steps "$TOTAL_ROUNDS"' in launcher
    assert "incompatible_checkpoint_quarantine.json" in launcher
    assert 'if [[ ! -f "$RUN_DIR/incompatible_checkpoint_quarantine.json" ]]' in launcher
