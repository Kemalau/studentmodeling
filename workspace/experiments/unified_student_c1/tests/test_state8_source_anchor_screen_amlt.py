from pathlib import Path

import yaml

from scripts.validate_state8_source_anchor_screen_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_source_anchor_screen_config_is_valid() -> None:
    validate(ROOT / "amlt/feeds_state8_source_anchor_screen_20260811.yaml")


def test_source_anchor_screen_config_supports_validated_recovery_subset(tmp_path) -> None:
    import yaml

    source = ROOT / "amlt/feeds_state8_source_anchor_screen_20260811.yaml"
    value = yaml.safe_load(source.read_text())
    value["jobs"] = [value["jobs"][1]]
    path = tmp_path / "subset.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    validate(path, {"state8-source-anchor-updater-mse1-seed3072"})


def test_validator_rejects_wrong_uai(tmp_path: Path) -> None:
    path = ROOT / "amlt/feeds_state8_source_anchor_screen_20260811.yaml"
    config = yaml.safe_load(path.read_text())
    config["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "wrong"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(config, sort_keys=False))
    try:
        validate(bad)
    except ValueError as error:
        assert "Environment" in str(error)
    else:
        raise AssertionError("wrong UAI was accepted")
