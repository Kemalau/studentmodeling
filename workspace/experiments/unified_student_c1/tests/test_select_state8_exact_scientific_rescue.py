from scripts.select_state8_exact_scientific_rescue import CANDIDATES


def test_rescue_candidates_have_unique_artifacts_and_finalfit_routes() -> None:
    assert set(CANDIDATES) == {
        "semantic_preserve", "causal_taskanchor", "retention_preserve"
    }
    assert len({value["job"] for value in CANDIDATES.values()}) == 3
    assert len({value["stable_root"] for value in CANDIDATES.values()}) == 3
    assert len({value["finalfit_variant"] for value in CANDIDATES.values()}) == 3
    assert all("diagnosis_first" in entrypoint for value in CANDIDATES.values() for entrypoint in value["entrypoints"])
