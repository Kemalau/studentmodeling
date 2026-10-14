from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_provenance_audit_is_fail_closed_on_full_diagnostic_recipe() -> None:
    text = (
        ROOT / "scripts/audit_layerwise_diagnostic_checkpoint_provenance.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        '"state_architecture": "layerwise"',
        '"state_token_alias": "<|vision_pad|>"',
        '"behavior_kd_weight": 0.0',
        '"direct_replay_weight": 0.0',
        '"causal_weight": 2.0',
        '"causal_objective_mode": "specificity_triplet"',
        '"diagnostic_start_global_position": 12000',
        '"diagnostic_stop_after_round": 3250',
        'len(real_order) != 259845',
        'len(causal_schedule) != 12992',
        'len(rank_rng_states) != 4',
        'positions != list(range(12000, 13000))',
        'len(set(example_ids)) != 1000',
        'tasks != collections.Counter(manifest.get("task_counts") or {})',
        'observed != receipt.get("sha256")',
        'Frozen diagnostic trainer SHA256 drift',
        'Open-T3 diagnostic entrypoint SHA256 drift',
        'Checkpoint mirror provenance drift',
    ):
        assert fragment in text
