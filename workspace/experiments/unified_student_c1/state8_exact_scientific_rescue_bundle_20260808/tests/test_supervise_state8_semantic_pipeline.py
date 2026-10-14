from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_owns_complete_state8_semantic_pipeline() -> None:
    text = (ROOT / "scripts/supervise_recurrent_unified_pipeline.sh").read_text()
    for fragment in (
        "state8_semantic_c4_g005",
        "state8_semantic_c4_g010",
        "state8_semantic_c4_g025",
        "state8_semantic_eedi_g005",
        "state8_semantic_eedi_g010",
        "state8_semantic_eedi_g025",
        "state8_semantic_eedi_g050",
        "state8_semantic_eedi_g025_single",
        "VALIDATOR=scripts/validate_state8_eedi_g025_single_amlt.py",
        "state8_semantic_training_progress",
        "watch_state8_unified_final_fit.sh",
        "watch_state8_final_eval.sh",
    ):
        assert fragment in text
