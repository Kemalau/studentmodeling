from pathlib import Path
import copy,yaml,pytest

from scripts.materialize_state8_eedi_same_question_final_eval_amlt import materialize
from scripts.validate_state8_eedi_same_question_final_eval_amlt import validate


ROOT=Path(__file__).resolve().parents[1]


def test_materializes_twenty_independent_a100_jobs(tmp_path: Path):
    out=tmp_path/"final.yaml"
    materialize(ROOT/"amlt/feeds_state8_final_eval_20260805.yaml",out)
    config=yaml.safe_load(out.read_text()); validate(config)
    assert len(config["jobs"])==20
    assert all(job["sku"]=="80G1-A100" for job in config["jobs"])
    assert any(job["name"]=="state8-final-eedi-same-question-c4-final150" for job in config["jobs"])
    assert all(job["submit_args"]["env"]["STATE8_FINAL_EVAL_RUN_DIR"].endswith("state8_eedi_same_question_unified_canonical_v1_20260811") for job in config["jobs"])


def test_validator_accepts_subset_wave_and_rejects_topology(tmp_path: Path):
    out=tmp_path/"final.yaml"; materialize(ROOT/"amlt/feeds_state8_final_eval_20260805.yaml",out)
    config=yaml.safe_load(out.read_text()); subset=copy.deepcopy(config); subset["jobs"]=subset["jobs"][:2]
    validate(subset,{job["name"] for job in subset["jobs"]})
    subset["jobs"][0]["sku"]="80G2-A100"
    with pytest.raises(ValueError,match="topology"): validate(subset,{job["name"] for job in subset["jobs"]})
