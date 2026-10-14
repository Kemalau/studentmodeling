from scripts.finalize_state8_source_anchor_answer_c4 import finalize


def test_joint_gate_requires_frozen_answer_c4_and_matching_artifact() -> None:
    artifact = {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}
    selection = {
        "stage": "state8_source_anchor_diagnostic_selection_v1",
        "diagnostic_gate_pass": True,
        "selected_for_answer_c4_confirmation": "full_mse1",
        "candidates": {
            "full_mse1": {
                "job": "state8-source-anchor-full-mse1-seed3071",
                "artifact_sha256": artifact,
                "checks": {},
                "floors": {},
            }
        },
    }
    cells = {name: {"flip_to_A_rate": value} for name, value in {
        "swap": 0.5, "random": 0.2, "samemal": 0.05, "noop": 0.0
    }.items()}
    answer = {
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "pair_seed": 42,
        "artifact_sha256": artifact,
        "summary": {
            "all_pairs_n": 150,
            "clean_n": 100,
            "clean": cells,
            "D_specific": 0.3,
            "scientific_gate_pass": True,
        },
    }
    result = finalize(selection, answer)
    assert result["joint_pilot_gate_pass"] is True
    answer["summary"]["scientific_gate_pass"] = False
    assert finalize(selection, answer)["joint_pilot_gate_pass"] is False
