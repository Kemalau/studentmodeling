from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/prepare_layerwise_consolidation_warm_start.py").read_text()
AUDIT = (ROOT / "scripts/audit_layerwise_diagnostic_warm_start.py").read_text()
RUNNER = (ROOT / "scripts/run_layerwise_task_retention_rehearsal.sh").read_text()


def test_warm_start_accepts_a_hash_bound_variable_prior_segment() -> None:
    assert "expected_manifest_sha256=prior_sha256" in SOURCE
    assert "prior_end != args.expected_round * args.expected_world_size" in SOURCE
    assert '"prior_manifest_sha256": prior_sha256' in SOURCE
    assert "expected_manifest_sha256: str" in AUDIT
    assert "== expected_manifest_sha256" in AUDIT
    assert '--expected-manifest-sha256 "$(sha256sum "$WARM_START_PRIOR_MANIFEST"' in RUNNER
