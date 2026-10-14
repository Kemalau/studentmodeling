from pathlib import Path

from scripts.validate_state8_exact_midpoint_c4_resilient_amlt import load, validate


def test_resilient_midpoint_c4_config_is_hash_bound_and_disjoint():
    config = load(Path("amlt/feeds_state8_exact_midpoint_c4_resilient_20260809.yaml"))
    validate(config)
    validate({**config, "jobs": [config["jobs"][0]]})


def test_answer_launcher_uses_remote_pair_level_resume():
    source = Path("scripts/run_state8_exact_midpoint_answer_c4_resumable.sh").read_text()
    assert '--progress-dir "$STATE8_EXACT_ANSWER_PROGRESS_DIR"' in source
    assert '--progress-key "$STATE8_EXACT_ANSWER_PROGRESS_KEY"' in source
    assert "--num-pairs 50 --allow-manifest-prefix --seed 42" in source


def test_diagnosis_launcher_uses_exact_diagnosis_first_contract():
    source = Path("scripts/run_state8_exact_midpoint_diagnosis_c4_shard.sh").read_text()
    assert "evaluate_recurrent_diagnosis_c4_diagnosis_first.py" in source
    assert '--pair-start "$PAIR_START"' in source
    assert '--pair-count "$PAIR_COUNT"' in source
