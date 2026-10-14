from scripts.select_state8_global_candidate import select


def common_checks(prefix: str = ""):
    if prefix == "curriculum":
        return {
            "T1_macro_f1": 0.65,
            "T2_math_equivalence": 0.35,
            "T3_eedi_semantic": 0.45,
            "T3_synthetic_cross_semantic": 0.60,
            "T3_synthetic_same_semantic": 0.60,
            "T3_schema": 1.0,
            "T5_joint_math_equivalence": 0.50,
            "T6_macro_f1": 0.40,
            "T6_positive_recall": 0.20,
            "malrule_diagnosis_C4_pass": True,
            "malrule_diagnosis_C4_D_specific": 0.30,
            "malrule_diagnosis_C4_swap_to_donor": 0.65,
            "malrule_diagnosis_C4_random_stable": 0.10,
            "malrule_diagnosis_C4_noop_retention": 0.85,
            "malrule_diagnosis_C4_same_rule_retention": 0.80,
            "eedi_same_question_C4_pass": True,
            "eedi_same_question_C4_D_specific": 0.35,
            "eedi_same_question_C4_swap_to_donor": 0.65,
            "eedi_same_question_C4_random_stable": 0.10,
            "eedi_same_question_C4_noop_retention": 0.85,
            "eedi_same_question_C4_same_rule_retention": 0.80,
        }
    value = common_checks("curriculum")
    mapping = {
        "T1": value.pop("T1_macro_f1"),
        "T2": value.pop("T2_math_equivalence"),
        "T5": value.pop("T5_joint_math_equivalence"),
    }
    value.update(mapping)
    return value


def selection(kind: str, score_delta: float = 0.0):
    curriculum = kind == "curriculum"
    checks = common_checks("curriculum" if curriculum else "repair")
    checks["T3_eedi_semantic"] += score_delta
    arm = (
        "curriculum_current_query_cp5000"
        if curriculum
        else "repair_semantic_binding_seed4201"
    )
    selected = {
        "arm": arm,
        "qualified": True,
        "score": 0.1,
        "checks": checks,
        "artifact_sha256": {
            "adapter_model": ("a" if curriculum else "c") * 64,
            "recurrent_state": ("b" if curriculum else "d") * 64,
        },
        "remote_root": (
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_diagnosis_first_eedi_current_query_curriculum_20260812/"
            "checkpoint-5000"
            if curriculum
            else "/mnt/amulet/studentmodeling/diagnostics/"
            "state8_stateonly_repair_seed4201_x"
        ),
        "finalfit_variant": (
            "same_question_curriculum_promoted"
            if curriculum
            else "same_question_repair_promoted"
        ),
    }
    return {
        "stage": (
            "state8_curriculum_checkpoint_joint_selection_v1"
            if curriculum
            else "state8_eedi_same_question_joint_diagnostic_selection_v1"
        ),
        "status": "complete",
        "pass": True,
        "selected_arm": arm,
        "selected": selected,
    }


def test_global_selector_recomputes_one_common_score():
    result = select(
        [
            ("curriculum", selection("curriculum")),
            ("proactive", selection("repair", score_delta=0.20)),
        ]
    )
    assert result["pass"] is True
    assert result["selected"]["source_selection"] == "proactive"
    assert result["selected"]["source_score_diagnostic_only"] == 0.1
    assert result["answer_C4_confirmation_pending"] is True


def test_global_selector_compares_both_proactive_checkpoint_lineages():
    cp3000 = selection("repair", score_delta=0.30)
    cp4000 = selection("repair", score_delta=0.10)
    cp3000["selected"]["arm"] = "repair_semantic_binding_seed6201"
    cp4000["selected"]["arm"] = "repair_semantic_binding_seed5201"
    result = select(
        [
            ("proactive_cp3000", cp3000),
            ("proactive_cp4000", cp4000),
        ]
    )
    assert result["pass"] is True
    assert result["selected"]["source_selection"] == "proactive_cp3000"


def test_global_selector_compares_control_robust_lineage() -> None:
    proactive = selection("repair", score_delta=0.10)
    robust = selection("repair", score_delta=0.35)
    robust["selected"]["arm"] = "repair_identity_controls_seed7202"
    robust["selected"]["remote_root"] = (
        "/mnt/amulet/studentmodeling/diagnostics/"
        "state8_stateonly_repair_seed7202_control_robust"
    )
    result = select(
        [
            ("proactive_cp4000", proactive),
            ("current_query_control_robust", robust),
        ]
    )
    assert result["pass"] is True
    assert result["selected"]["source_selection"] == "current_query_control_robust"


def test_global_selector_reapplies_common_hard_gates():
    broken = selection("repair", score_delta=0.50)
    broken["selected"]["checks"]["T6_positive_recall"] = 0.0
    result = select([("proactive", broken)])
    assert result["pass"] is False
    assert "T5_positive_recall" in result["candidates"][0]["failed_checks"]
    assert result["task_taxonomy"]["legacy_to_public"]["T6"] == "T5"


def test_global_selector_accepts_hash_bound_repair_checkpoint_layout():
    checkpoint = selection("repair", score_delta=0.25)
    checkpoint["selected"].update(
        arm="repair_checkpoint_semantic_binding_seed6201_cp500",
        artifact_layout="repair_checkpoint_snapshot",
        finalfit_variant="same_question_repair_checkpoint_promoted",
        remote_root=(
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_curriculum_repairs_20260812/"
            "seed6201_semantic_binding/checkpoint-500"
        ),
    )
    result = select([("repair_checkpoint_cp3000", checkpoint)])
    assert result["pass"] is True
    assert result["selected"]["artifact_layout"] == "repair_checkpoint_snapshot"


def test_global_selector_compares_causal_teacher_binding_lineage():
    checkpoint = selection("repair", score_delta=0.30)
    checkpoint["selected"].update(
        arm="repair_checkpoint_causal_teacher_high_seed8202_cp100",
        artifact_layout="repair_checkpoint_snapshot",
        finalfit_variant="same_question_repair_checkpoint_promoted",
        remote_root=(
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_curriculum_repairs_20260812/"
            "seed8202_causal_teacher_high/checkpoint-100"
        ),
    )
    result = select([("causal_teacher_binding", checkpoint)])
    assert result["pass"] is True
    assert result["selected"]["source_selection"] == "causal_teacher_binding"


def test_failed_source_is_audited_but_not_ranked():
    failed = selection("repair")
    failed["pass"] = False
    failed["selected"] = None
    result = select(
        [("curriculum", selection("curriculum")), ("repair", failed)]
    )
    assert result["pass"] is True
    assert result["skipped_sources"][0]["source"] == "repair"
