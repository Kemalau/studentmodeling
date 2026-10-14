from scripts.task_taxonomy_v2 import (
    APPENDIX_TASKS,
    HEADLINE_TASKS,
    LEGACY_TO_PUBLIC,
    TASK_KEY_TO_PUBLIC,
    taxonomy_receipt,
)


def test_v2_numbering_is_contiguous_and_backward_auditable() -> None:
    assert HEADLINE_TASKS == ("T1", "T2", "T3", "T4", "T5")
    assert APPENDIX_TASKS == ("T6",)
    assert LEGACY_TO_PUBLIC == {
        "T1": "T1",
        "T2": "T2",
        "T3": "T3",
        "T5": "T4",
        "T6": "T5",
        "T4": "T6",
    }
    assert TASK_KEY_TO_PUBLIC["predict_prospective_error"] == "T4"
    assert TASK_KEY_TO_PUBLIC["predict_error_persistence"] == "T5"
    assert TASK_KEY_TO_PUBLIC["estimate_mastery"] == "T6"
    assert taxonomy_receipt()["internal_keys_stable_for_artifact_compatibility"]
