import json

from scripts.build_t1_t2_paired_addone import event_key


def test_event_key_matches_t1_and_t2_versions():
    t1 = {"example_id": "fullmix:x:predict_correctness:s:1", "metadata": {"original_example_id": "x:predict_correctness:s:1"}}
    t2 = {"example_id": "fullmix:x:predict_response:s:1", "metadata": {"original_example_id": "x:predict_response:s:1"}}
    assert event_key(t1) == event_key(t2) == "x:TASK:s:1"


def test_event_key_preserves_nonpaired_malrule_task_name():
    row = {"example_id": "fullmix:malrule:malrule_next_response", "metadata": {"original_example_id": "malrule:malrule_next_response"}}
    assert event_key(row) == "malrule:malrule_next_response"
