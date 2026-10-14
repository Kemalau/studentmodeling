import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

from scripts.materialize_state8_eedi_same_question_repair_branches import materialize
from scripts.validate_state8_eedi_same_question_repair_branches_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "amlt/feeds_state8_eedi_same_question_causal_repair_20260811.yaml"


def test_materializes_hash_bound_parallel_repair_jobs(tmp_path: Path):
    results = tmp_path / "results"
    branches = []
    for index, (name, arm, step, training) in enumerate((
        ("semantic_binding", "updater", 20, "full"),
        ("specificity_triplet", "full", 20, "updater"),
        ("identity_controls", "updater", 100, "updater"),
    )):
        receipt = results / f"state8_eedi_same_question_{arm}_checkpoint{step}_screen_20260811/checkpoint_completion.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"status":"complete","step":step,"sha256":{
            "adapter_model": chr(97 + index) * 64,
            "recurrent_state": chr(100 + index) * 64,
        }}))
        branches.append({
            "name": name, "source": {"arm": arm, "checkpoint_step": step},
            "training_arm": training,
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS":"150",
                "STATE8_CAUSAL_REPEAT_FRACTION":"0.75",
                "STATE8_CAUSAL_WEIGHT":"3.0",
                "STATE8_CAUSAL_MARGIN":"0.75",
                "STATE8_EEDI_GEOMETRY_WEIGHT":"0.05",
            },
        })
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "stage":"state8_eedi_same_question_failed_checkpoint_repair_plan_v1",
        "scientific_floors_unchanged":True,
        "raw_history_visible_to_decoder":False,
        "branches":branches,
    }))
    output = tmp_path / "bound.yaml"
    materialize(TEMPLATE, plan, results, output)
    validate(output)
    text = output.read_text()
    assert text.count("sku: 80G1-A100") == 3
    assert "rankfun_aml" in text
    assert "msrcommonacr.azurecr.io/ml-common-py3.12:latest" not in text


def test_repair_branch_renders_disjoint_checkpoint_warm_start(tmp_path: Path):
    env = {
        **os.environ,
        "STATE8_RENDER_ONLY": "1",
        "REPAIR_BRANCH": "semantic_binding",
        "REPAIR_SEED": "3201",
        "SOURCE_CHECKPOINT_ARM": "updater",
        "SOURCE_CHECKPOINT_STEP": "20",
        "SOURCE_CHECKPOINT_ROOT": (
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_eedi_same_question_20260811/updater/checkpoint-20"
        ),
        "SOURCE_ANCHOR_MODE": "full_mse1",
        "EXPECTED_STUDENT_ADAPTER_SHA256": "a" * 64,
        "EXPECTED_STUDENT_STATE_SHA256": "b" * 64,
        "STATE8_EEDI_CAUSAL_TWINS": "200",
        "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
        "STATE8_CAUSAL_WEIGHT": "1.5",
        "STATE8_CAUSAL_MARGIN": "0.5",
        "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
        "REPAIR_REMOTE_ROOT": (
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed3201_taskanchor_source_anchor_"
            "full_mse1_eedi_same_question_v1_semantic_binding_seed3201"
        ),
    }
    output = Path("/tmp/run_state8_eedi_same_question_causal_repair.final.sh")
    output.unlink(missing_ok=True)
    subprocess.run(
        ["bash", "scripts/run_state8_eedi_same_question_repair_branch.sh"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    rendered = output.read_text(encoding="utf-8")
    assert "--twins 200 --seed 20260811 --offset 0" in rendered
    assert "--twins 25 --seed 20260811 --offset 200" in rendered
    assert '--exclude-jsonl "$EEDI_CAUSAL_TRAIN"' in rendered
    assert "assert not train_entities & valid_entities" in rendered
    assert (
        "import json,os,sys\n"
        "train,valid,merged=(json.load(open(path)) for path in sys.argv[1:])"
        in rendered
    )
    assert "expected_pairs=250 if" in rendered and "else 900" in rendered
    assert (
        "test \"$(sha256sum scripts/evaluate_recurrent_formal_protocols.py | "
        "cut -d' ' -f1)\" = "
        "208edcc8c623101667844910ad3e85c79d82491a08e370acc468ad617ab62522"
        in rendered
    )
    outer = Path("/tmp/run_state8_eedi_same_question_source_anchor.rendered.sh").read_text()
    assert "STUDENT_MODEL_DIR=\"$STUDENT_ROOT/adapter\"" in outer
    assert "STUDENT_RECEIPT=\"$STUDENT_ROOT/checkpoint_completion.json\"" in outer
    subprocess.run(["bash", "-n", str(output)], check=True)


def test_materializes_curriculum_checkpoint_repair_source(tmp_path: Path):
    results = tmp_path / "results"
    receipt = (
        results
        / "state8_diagnosis_first_eedi_current_query_curriculum_20260812"
        / "milestones/checkpoint-2000.completion.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "status": "complete",
        "step": 2000,
        "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "stage": "state8_curriculum_failed_checkpoint_repair_plan_v1",
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "branches": [{
            "name": "semantic_binding",
            "source": {
                "family": "curriculum",
                "arm": "current_query",
                "checkpoint_step": 2000,
            },
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "200",
                "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
                "STATE8_CAUSAL_WEIGHT": "1.5",
                "STATE8_CAUSAL_MARGIN": "0.5",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
            },
        }],
    }))
    output = tmp_path / "bound.yaml"
    materialize(TEMPLATE, plan, results, output)
    validate(output)
    value = __import__("yaml").safe_load(output.read_text())
    env = value["jobs"][0]["submit_args"]["env"]
    assert env["SOURCE_CHECKPOINT_ARM"] == "curriculum_current_query"
    assert env["SOURCE_CHECKPOINT_STEP"] == "2000"
    assert env["SOURCE_CHECKPOINT_ROOT"].endswith(
        "state8_diagnosis_first_eedi_current_query_curriculum_20260812/checkpoint-2000"
    )


def test_proactive_curriculum_seed_namespace_is_disjoint(tmp_path: Path):
    results = tmp_path / "results"
    receipt = (
        results
        / "state8_diagnosis_first_eedi_current_query_curriculum_20260812"
        / "milestones/checkpoint-4000.completion.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "status": "complete", "step": 4000,
        "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }))
    plan = tmp_path / "proactive.json"
    plan.write_text(json.dumps({
        "stage": "state8_curriculum_failed_checkpoint_repair_plan_v1",
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "seed_base": 5201,
        "branches": [{
            "name": "semantic_binding",
            "source": {"family": "curriculum", "arm": "current_query", "checkpoint_step": 4000},
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "200",
                "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
                "STATE8_CAUSAL_WEIGHT": "1.5",
                "STATE8_CAUSAL_MARGIN": "0.5",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
            },
        }],
    }))
    output = tmp_path / "proactive.yaml"
    materialize(TEMPLATE, plan, results, output)
    validate(output)
    env = yaml.safe_load(output.read_text())["jobs"][0]["submit_args"]["env"]
    assert env["REPAIR_SEED"] == "5201"
    assert "seed5201" in env["REPAIR_REMOTE_ROOT"]


def test_validator_accepts_exact_amlt_dump_target_shape(tmp_path: Path):
    results = tmp_path / "results"
    receipt = (
        results
        / "state8_diagnosis_first_eedi_current_query_curriculum_20260812"
        / "milestones/checkpoint-3000.completion.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "status": "complete", "step": 3000,
        "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "stage": "state8_curriculum_failed_checkpoint_repair_plan_v1",
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "seed_base": 6201,
        "branches": [{
            "name": "semantic_binding",
            "source": {"family": "curriculum", "arm": "current_query", "checkpoint_step": 3000},
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "200",
                "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
                "STATE8_CAUSAL_WEIGHT": "1.5",
                "STATE8_CAUSAL_MARGIN": "0.5",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
            },
        }],
    }))
    output = tmp_path / "dumped.yaml"
    materialize(TEMPLATE, plan, results, output)
    config = yaml.safe_load(output.read_text())
    config["target"] = {
        "name": "Feeds",
        "workspace_name": "CS-NewsAndFeeds-Singularity",
        "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
    }
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    validate(output)


def test_validator_accepts_amlt_dump_sing_service(tmp_path: Path):
    results = tmp_path / "results"
    receipt = (
        results
        / "state8_diagnosis_first_eedi_current_query_curriculum_20260812"
        / "milestones/checkpoint-3000.completion.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "status": "complete", "step": 3000,
        "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "stage": "state8_curriculum_failed_checkpoint_repair_plan_v1",
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "seed_base": 8201,
        "branches": [{
            "name": "semantic_binding",
            "source": {"family": "curriculum", "arm": "current_query", "checkpoint_step": 3000},
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "200",
                "STATE8_CAUSAL_REPEAT_FRACTION": "0.75",
                "STATE8_CAUSAL_WEIGHT": "2.0",
                "STATE8_CAUSAL_MARGIN": "0.5",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10"
            },
        }],
    }))
    output = tmp_path / "dumped-sing.yaml"
    materialize(TEMPLATE, plan, results, output)
    config = yaml.safe_load(output.read_text())
    config["target"] = {
        "service": "sing",
        "name": "Feeds",
        "workspace_name": "CS-NewsAndFeeds-Singularity",
        "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
    }
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    validate(output)


def test_validator_rejects_other_dump_workspace(tmp_path: Path):
    source = tmp_path / "config.yaml"
    value = yaml.safe_load(TEMPLATE.read_text())
    value["target"] = {
        "name": "Feeds",
        "workspace_name": "wrong-workspace",
        "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
    }
    source.write_text(yaml.safe_dump(value, sort_keys=False))
    import pytest
    with pytest.raises(ValueError, match="Target drift"):
        validate(source)


def test_validator_rejects_stale_runner_binding(tmp_path: Path):
    results = tmp_path / "results"
    receipt = (
        results
        / "state8_diagnosis_first_eedi_current_query_curriculum_20260812"
        / "milestones/checkpoint-3000.completion.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "status": "complete", "step": 3000,
        "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "stage": "state8_curriculum_failed_checkpoint_repair_plan_v1",
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "seed_base": 7201,
        "branches": [{
            "name": "semantic_binding",
            "source": {"family": "curriculum", "arm": "current_query", "checkpoint_step": 3000},
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "200",
                "STATE8_CAUSAL_REPEAT_FRACTION": "1.0",
                "STATE8_CAUSAL_WEIGHT": "1.5",
                "STATE8_CAUSAL_MARGIN": "0.5",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
            },
        }],
    }))
    output = tmp_path / "bound.yaml"
    materialize(TEMPLATE, plan, results, output)
    value = yaml.safe_load(output.read_text())
    value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
        hashlib.sha256(
            (ROOT / "scripts/run_state8_eedi_same_question_repair_branch.sh").read_bytes()
        ).hexdigest(),
        "0" * 64,
    )
    output.write_text(yaml.safe_dump(value, sort_keys=False))
    import pytest
    with pytest.raises(ValueError, match="Runner binding drift"):
        validate(output)


def test_materializes_causal_teacher_binding_pair(tmp_path: Path):
    results = tmp_path / "results"
    receipt = (
        results
        / "state8_diagnosis_first_eedi_current_query_curriculum_20260812"
        / "milestones/checkpoint-3000.completion.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "status": "complete", "step": 3000,
        "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "stage": "state8_curriculum_failed_checkpoint_repair_plan_v1",
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "seed_base": 8201,
        "branches": [{
            "name": "causal_teacher_low",
            "source": {
                "family": "curriculum", "arm": "current_query",
                "checkpoint_step": 3000,
            },
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "200",
                "STATE8_CAUSAL_REPEAT_FRACTION": "0.75",
                "STATE8_CAUSAL_WEIGHT": "2.0",
                "STATE8_CAUSAL_MARGIN": "0.5",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
                "STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT": "0.15",
                "STATE8_CAUSAL_TEACHER_COSINE_WEIGHT": "1.0",
                "STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT": "1.0",
            },
        }],
    }))
    output = tmp_path / "causal_teacher.yaml"
    materialize(TEMPLATE, plan, results, output)
    validate(output)
    env = yaml.safe_load(output.read_text())["jobs"][0]["submit_args"]["env"]
    assert env["REPAIR_BRANCH"] == "causal_teacher_low"
    assert env["STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT"] == "0.15"
    assert env["REPAIR_SEED"] == "8201"


def test_materializes_early_label_binding_pair(tmp_path: Path):
    results = tmp_path / "results"
    receipt = (
        results
        / "state8_diagnosis_first_eedi_current_query_curriculum_20260812"
        / "milestones/checkpoint-3000.completion.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "status": "complete", "step": 3000,
        "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "stage": "state8_curriculum_failed_checkpoint_repair_plan_v1",
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "seed_base": 8301,
        "branches": [{
            "name": "early_label_low",
            "source": {
                "family": "curriculum", "arm": "current_query",
                "checkpoint_step": 3000,
            },
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "200",
                "STATE8_CAUSAL_REPEAT_FRACTION": "0.75",
                "STATE8_CAUSAL_WEIGHT": "0.5",
                "STATE8_CAUSAL_MARGIN": "0.5",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
                "STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT": "0.15",
                "STATE8_CAUSAL_TEACHER_COSINE_WEIGHT": "1.0",
                "STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT": "1.0",
                "STATE8_EARLY_LABEL_BINDING_WEIGHT": "0.25",
            },
        }],
    }))
    output = tmp_path / "early-label.yaml"
    materialize(TEMPLATE, plan, results, output)
    validate(output)
    env = yaml.safe_load(output.read_text())["jobs"][0]["submit_args"]["env"]
    assert env["REPAIR_BRANCH"] == "early_label_low"
    assert env["STATE8_EARLY_LABEL_BINDING_WEIGHT"] == "0.25"
    assert env["STATE8_CAUSAL_WEIGHT"] == "0.5"


def test_materializes_control_robust_cp5000_repair(tmp_path: Path):
    results = tmp_path / "results"
    receipt = (
        results
        / "state8_diagnosis_first_eedi_current_query_curriculum_20260812"
        / "milestones/checkpoint-5000.completion.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "status": "complete", "step": 5000,
        "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
    }))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "stage": "state8_curriculum_failed_checkpoint_repair_plan_v1",
        "scientific_floors_unchanged": True,
        "raw_history_visible_to_decoder": False,
        "seed_base": 7201,
        "branches": [{
            "name": "semantic_binding",
            "source": {
                "family": "curriculum", "arm": "current_query",
                "checkpoint_step": 5000,
            },
            "training_arm": "full",
            "overrides": {
                "STATE8_EEDI_CAUSAL_TWINS": "200",
                "STATE8_CAUSAL_REPEAT_FRACTION": "1.0",
                "STATE8_CAUSAL_WEIGHT": "2.0",
                "STATE8_CAUSAL_MARGIN": "0.75",
                "STATE8_EEDI_GEOMETRY_WEIGHT": "0.10",
                "STATE8_CONTROL_RANDOM_SAMPLES": "4",
                "STATE8_CONTROL_RANDOM_WEIGHT": "0.25",
                "STATE8_CONTROL_SAME_RULE_WEIGHT": "0.50",
            },
        }],
    }))
    output = tmp_path / "bound.yaml"
    materialize(TEMPLATE, plan, results, output)
    validate(output)
    env = yaml.safe_load(output.read_text())["jobs"][0]["submit_args"]["env"]
    assert env["SOURCE_CHECKPOINT_STEP"] == "5000"
    assert env["STATE8_CONTROL_RANDOM_SAMPLES"] == "4"
    assert env["STATE8_CONTROL_RANDOM_WEIGHT"] == "0.25"
    assert env["STATE8_CONTROL_SAME_RULE_WEIGHT"] == "0.50"
