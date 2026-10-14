from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_source_anchor_screen_is_three_arm_hash_bound_and_strict_state_only() -> None:
    text = (ROOT / "scripts/run_state8_source_anchor_screen.sh").read_text()
    for fragment in (
        "full_mse1",
        "updater_mse1",
        "full_mse2",
        "STATE8_TRAJECTORY_RELATIVE_MSE_WEIGHT",
        "train_state8_diagnosis_first_source_anchored_repair.py",
        "state8_diagnosis_first_full_history_teacher_v1",
        "STATE8_LAYERWISE_TRANSITION_ONLY=1",
        "STATE8_UPDATER_ONLY=1",
        "STATE8_FORMAL_SCOPE=unified",
        "STATE8_CAUSAL_DIAGNOSIS_ONLY=0",
        "source_hash=2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7",
        "EXPECTED_STUDENT_ADAPTER_SHA256",
        "EXPECTED_STUDENT_STATE_SHA256",
    ):
        assert fragment in text
