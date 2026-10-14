from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_watcher_recovers_preserves_and_uses_authoritative_semantic_judge() -> None:
    text = (ROOT / "scripts/watch_state8_taskanchor_milestone_screen.sh").read_text()
    for fragment in (
        "amlt results download",
        "amlt resume",
        "validate_state8_taskanchor_milestone_screen_amlt.py",
        "--allow-job-subset",
        "--judge-backend trapi",
        "gpt-4o_2024-11-20",
        "summarize_state8_diagnosis_c4.py",
        "selected_for_full_confirmation",
        "requires full confirmation",
    ):
        assert fragment in text
