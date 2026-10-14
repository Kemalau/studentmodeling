from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_preflight_is_read_only_and_provenance_bound() -> None:
    script = (ROOT / "scripts/watch_layerwise_equal_state_path_local_preflight.sh").read_text()
    required = (
        'checkpoint="$state_dir/checkpoint-round-$checkpoint_round"',
        'x.get("status") == "complete" and x.get("round") == int(sys.argv[2])',
        'x.get("world_size") == 4',
        'assert observed == receipt["sha256"]',
        "--state-architecture layerwise",
        "--max-samples-per-group 10",
        "--num-pairs 50",
        '"selection": "ordered_prefix"',
        '"parent_manifest_sha256"',
        '"diagnostic_only": True',
        '"not_a_formal_gate": True',
        '"loss_profile": sys.argv[5]',
        "raw_history_visible_to_decoder",
    )
    # The final fragment is provided by the frozen evaluators rather than the
    # watcher, so the watcher must never claim a formal pass itself.
    assert all(fragment in script for fragment in required[:-1])
    forbidden = ("amlt run", "optimizer.step", "--train-jsonl", "scientific_gate_pass")
    assert not any(fragment in script for fragment in forbidden)


def test_local_preflight_uses_fixed_storage_and_alias() -> None:
    script = (ROOT / "scripts/watch_layerwise_equal_state_path_local_preflight.sh").read_text()
    assert "csnewsandfeeds4150361735" in script
    assert "rankfun_aml" not in script
    assert "<|vision_pad|>" in script
    assert "AZURE_CONFIG_DIR" in script


def test_local_preflight_can_audit_state_path_or_state_kd_without_changing_protocol() -> None:
    script = (ROOT / "scripts/watch_layerwise_equal_state_path_local_preflight.sh").read_text()
    assert 'state_slug="${STATE_SLUG:-equal_state_path}"' in script
    assert 'loss_profile="${LOSS_PROFILE_EXPECTED:-state_path}"' in script
    assert 'equal_t3_t6/$loss_profile/lr5e-7' in script
    assert 'checkpoint_round="${CHECKPOINT_ROUND:-3250}"' in script
    assert 'prefix="${REMOTE_PREFIX:-' in script
    assert '"loss_profile": "${loss_profile}"' not in script


def test_local_preflights_serialize_on_the_single_local_gpu() -> None:
    script = (ROOT / "scripts/watch_layerwise_equal_state_path_local_preflight.sh").read_text()
    assert 'results/local_gpu_readonly_preflight.lock' in script
    assert 'while ! flock -n 10' in script
    assert 'nvidia-smi --query-compute-apps=pid' in script
    assert 'heartbeat' in script
