from scripts.audit_state8_source_anchor_checkpoint import ARMS


def test_same_question_checkpoint_arms_are_isolated_and_query_routed():
    updater = ARMS["eedi_same_question_updater_mse1"]
    full = ARMS["eedi_same_question_full_mse1"]
    assert updater["seed"] == 3072 and updater["transition_only"] is True
    assert full["seed"] == 3071 and full["transition_only"] is False
    assert updater["tag"] != full["tag"]
    assert updater["current_query"] == full["current_query"] == "eedi_only"
