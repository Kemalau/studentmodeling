import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.materialize_state8_eedi_same_question_finalfit_world2_amlt import materialize
from scripts.validate_state8_eedi_same_question_finalfit_world2_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "amlt/feeds_state8_eedi_same_question_final_fit_world2_20260811.yaml"


def promotion(tmp_path: Path, variant="same_question_updater_mse1_seed3072") -> Path:
    path = tmp_path / "promotion.json"
    path.write_text(json.dumps({
        "stage":"state8_eedi_same_question_joint_promotion_v1", "pass":True,
        "selected":{
            "qualified":True, "finalfit_variant":variant, "answer_C4_pass":True,
            "checks":{"malrule_diagnosis_C4_pass":True,"eedi_same_question_C4_pass":True},
        },
    }))
    return path


def test_materializes_one_promoted_world2_job(tmp_path: Path):
    output = tmp_path / "bound.yaml"
    materialize(TEMPLATE, promotion(tmp_path), output)
    config = yaml.safe_load(output.read_text())
    validate(config)
    assert len(config["jobs"]) == 1
    assert config["jobs"][0]["sku"] == "80G2-A100"
    assert config["jobs"][0]["submit_args"]["env"]["FINALFIT_VARIANT"].startswith("same_question_")
    command = config["jobs"][0]["command"]
    assert "736d06fd7596707c9a669f9bff0a649d34134c40814fc5c88f77b268e61d1615" in command
    assert "8d025cb0301b894b85c1e2eac34bc3f318dc269fc76bd581b8aaa2bc887cc59f" in command


def test_finalfit_watcher_requires_control_robust_receipt() -> None:
    source = (ROOT / "scripts/watch_state8_eedi_same_question_finalfit_world2.sh").read_text()
    assert "fit['control_robust_objective'] is True" in source
    assert "fit['control_random_samples'])==4" in source
    assert "fit['control_random_weight'])==0.25" in source
    assert "fit['control_same_rule_weight'])==0.50" in source
    assert "fit['causal_teacher_trajectory_weight']" in source
    assert "fit['early_label_binding_weight']" in source
    assert "binding_constraints_inherited_from_promotion" in source


def test_validator_rejects_topology_or_uai_drift(tmp_path: Path):
    output = tmp_path / "bound.yaml"
    materialize(TEMPLATE, promotion(tmp_path), output)
    config = yaml.safe_load(output.read_text())
    bad = copy.deepcopy(config); bad["jobs"][0]["sku"]="80G1-A100"
    with pytest.raises(ValueError, match="Topology drift"): validate(bad)
    bad = copy.deepcopy(config); bad["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"]="other"
    with pytest.raises(ValueError, match="UAI drift"): validate(bad)
