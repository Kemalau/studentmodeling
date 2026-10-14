from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_round35_probe_uses_trapi_and_audits_collapse() -> None:
    source = (ROOT / "scripts/watch_state8_taxonomy_consolidation_round35_t3.sh").read_text()
    for fragment in (
        "judge-backend trapi",
        "generation_rows']==semantic['judged_rows']==75",
        "eedi_unique_output_rate",
        "eedi_most_common_output_rate",
        "exact_match_authoritative':False",
        "needs_attention",
    ):
        assert fragment in source
