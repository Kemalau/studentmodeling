from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_c4_real_random_dose import summarize  # noqa: E402


def test_summary_uses_native_clean_subset_and_paired_control():
    records = []
    for pair, native in [(1, "stay_B"), (2, "flip_to_A")]:
        categories = {
            "none": native, "noop": "stay_B", "dose_0p25": "stay_B",
            "dose_0p5": "stay_B", "dose_0p75": "flip_to_A", "swap": "flip_to_A",
            "random": "stay_B", "samemal": "stay_B", "random_real": "stay_B",
        }
        for index, (condition, category) in enumerate(categories.items()):
            records.append({
                "pair_idx": pair, "condition": condition, "category": category,
                "score_A": float(index), "score_B": 0.0, "p_A": index / 10,
                "donor_target": "A", "real_random_target_on_recipient_query": "C",
            })
    payload = {
        "records": records, "num_pairs": 2, "real_random_seed": 1,
        "real_random_selection": "test", "dose_formula": "test",
    }
    result = summarize(payload, bootstrap_samples=100)
    assert result["clean_n"] == 1
    assert result["real_random"]["directed_minus_real_random"] == 1.0
    assert result["dose_response"]["flip_rate_nondecreasing"]
