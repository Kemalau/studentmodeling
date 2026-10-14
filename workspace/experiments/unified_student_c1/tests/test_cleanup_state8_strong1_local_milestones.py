from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cleanup_preserves_remote_authority_and_waits_for_consumers() -> None:
    source = (
        ROOT / "scripts/cleanup_state8_strong1_local_milestones.sh"
    ).read_text()
    for fragment in (
        "steps=(250 1000 2000 3000 4000 5000)",
        'step" == 250 && ! -f "$t3/complete"',
        "remote/checkpoint_completion.json",
        "remote/recurrent_state.pt",
        "a['sha256']==s['artifact_sha256']",
        "local_metrics_and_hash_receipts_preserved",
    ):
        assert fragment in source
    assert "az storage blob delete" not in source
    assert "rm -rf" not in source
