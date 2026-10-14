from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_recovered_summary_requires_all_frozen_evidence() -> None:
    text = (
        ROOT / "scripts/summarize_layerwise_local_recovered_candidate.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        '"equal_t3_t6_open_t3__state_path"',
        'provenance.get("optimizer_present") is True',
        'provenance.get("scheduler_present") is True',
        'provenance.get("full_real_order_rows") == 259845',
        'provenance.get("causal_schedule_size") == 12992',
        '"closed_layerwise":',
        '"C4": bool(c4_summary.get("scientific_gate_pass"))',
        '"formal_screen": bool(formal.get("diagnostic_gate_pass"))',
        '"T3_semantic": bool(t3.get("pass"))',
        't6_macro / T6_REFERENCE_MACRO_F1 >= T6_MIN_RATIO',
        '"open_t3_training_view": provenance_pass',
        '"all_gates_pass": all(checks.values())',
    ):
        assert fragment in text
