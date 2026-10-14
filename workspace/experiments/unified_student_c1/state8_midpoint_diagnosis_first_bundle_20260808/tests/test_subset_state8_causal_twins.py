import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from subset_state8_causal_twins import select  # noqa: E402


def row(twin: int, role: str) -> dict:
    return {
        "pair_id": f"old-{twin}-{role}",
        "same_rule_recipient_id": f"recipient-{twin}",
        "same_rule_control_id": f"control-{twin}",
        "donor_id": f"donor-{twin}",
        "native_control_role": role,
        "family": "fractions",
    }


def test_subset_preserves_complete_twins_and_renumbers() -> None:
    rows = [
        row(index, role)
        for index in range(4)
        for role in ("recipient_native", "same_rule_native")
    ]
    selected = select(rows, twins=2, seed=42)
    assert len(selected) == 4
    assert [item["pair_id"] for item in selected] == [
        "state8-qualified-causal:0000000",
        "state8-qualified-causal:0000001",
        "state8-qualified-causal:0000002",
        "state8-qualified-causal:0000003",
    ]
    assert [item["native_control_role"] for item in selected] == [
        "recipient_native",
        "same_rule_native",
        "recipient_native",
        "same_rule_native",
    ]
