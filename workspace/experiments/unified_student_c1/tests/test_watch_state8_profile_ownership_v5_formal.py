from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts/watch_state8_profile_ownership_v5_formal.sh"


def test_v5_formal_watcher_is_fail_closed_and_monitor_only() -> None:
    source = WATCHER.read_text(encoding="utf-8")
    assert "state8-profile-ownership-v5-formal-20260822" in source
    assert "success={'pass','passed'}" in source
    assert "failure={'failed','fail','killed','cancelled','canceled','stopped'}" in source
    assert "all(value in success | failure for value in states)" in source
    assert "amlt run" not in source
    assert "device-code" not in source
    assert "use-device-code" not in source


def test_v5_formal_watcher_downloads_auditable_artifacts() -> None:
    source = WATCHER.read_text(encoding="utf-8")
    for artifact in (
        "*.json",
        "*.npz",
        "readable_profiles.jsonl",
        "fixed_query_transport.jsonl",
        "adapter_model.safetensors",
        "recurrent_state.pt",
    ):
        assert artifact in source
    assert "audit_state8_future_profile_v2.py" in source
    assert "downloaded_artifact_hashes_match_metrics" in source
    assert "independent_audit_exit_code" in source
    assert "completion_ready" in source
    assert "summarize_state8_future_profile_experiment.py" in source
    assert "final_experiment_summary.json" in source


def test_v5_formal_watcher_retries_download_and_collects_failed_receipts() -> None:
    source = WATCHER.read_text(encoding="utf-8")
    assert "for attempt in $(seq 1 12)" in source
    assert "download.partial" in source
    assert "download.failed-job.log" in source
    assert "--status pass" in source
