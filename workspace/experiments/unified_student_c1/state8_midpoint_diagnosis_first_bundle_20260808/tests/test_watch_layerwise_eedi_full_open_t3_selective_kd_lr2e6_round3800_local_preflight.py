from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "scripts/watch_layerwise_eedi_full_open_t3_selective_kd_lr2e6_round3800_local_preflight.sh"
).read_text(encoding="utf-8")


def test_wrapper_pins_the_audited_round3800_selective_kd_artifact() -> None:
    for fragment in (
        "STATE_SLUG=eedi_full_open_t3_selective_kd_lr_scan",
        "LOSS_PROFILE_EXPECTED=state_kd",
        "ARTIFACT_LABEL=open-t3-selective-kd-lr2e6",
        "CHECKPOINT_ROUND=3800",
        "LOCAL_PREFLIGHT_SUBDIR=local_lr2e6_round3800_preflight",
        "eedi_full_open_t3/state_kd/lr2e-6_start12000_stop4250_",
        "/checkpoint-round-3800",
        "exec bash scripts/watch_layerwise_equal_state_path_local_preflight.sh",
    ):
        assert fragment in SOURCE
