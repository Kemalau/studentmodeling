import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_exact_rescue_is_hash_bound_and_strict_state_only() -> None:
    source = (ROOT / "scripts/run_state8_exact_scientific_rescue.sh").read_text()
    for fragment in (
        "EXPECTED_STUDENT_ADAPTER_SHA256",
        "EXPECTED_STUDENT_STATE_SHA256",
        "EXACT_CURRICULUM_ROOT",
        "diagnosis_first_teacher_curriculum.json",
        "state8_stateonly_repair_pilot_publish_v1",
        "train_state8_diagnosis_first_task_anchored_causal_repair.py",
        "semantic_preserve",
        "causal_taskanchor",
        "diagnosis_first_task_anchored_causal_repair.json",
        "run_state8_stateonly_repair_ab.sh",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
        "--num-pairs 150",
    ):
        assert fragment in source


def test_exact_rescue_renders_a_syntax_valid_composed_launcher(
    tmp_path: Path,
) -> None:
    """Exercise the embedded renderer, not just static source fragments."""
    env = {
        **os.environ,
        "VENV_ROOT": str(tmp_path / "venv"),
        "EXACT_SOURCE_ROOT": (
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed65_trajectory01_"
            "exact_teacher_curriculum_c4_seed65_v1"
        ),
        "EXACT_CURRICULUM_ROOT": (
            "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_qualified_teacher_curriculum_highkd_v1_immutable_"
            "taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4"
        ),
        "EXPECTED_STUDENT_ADAPTER_SHA256": "a" * 64,
        "EXPECTED_STUDENT_STATE_SHA256": "b" * 64,
        "STATE8_COMPOSED_ENTRYPOINT_SHA256": (
            "eb9837127af9fe3761f6fa57b78a3e9ba3373a223706f0a4921749eab61e35a3"
        ),
        "STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256": (
            "7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216"
        ),
        "STATE8_RESCUE_MODE": "semantic_preserve",
        "AMLT_OUTPUT_DIR": str(tmp_path / "output"),
    }
    rendered = Path("/tmp/run_state8_exact_scientific_rescue.rendered.sh")
    rendered.unlink(missing_ok=True)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_state8_exact_scientific_rescue.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert rendered.is_file()
    assert "syntax error" not in result.stderr
    assert "\n+      ||" not in rendered.read_text()
    subprocess.run(["bash", "-n", str(rendered)], check=True)
