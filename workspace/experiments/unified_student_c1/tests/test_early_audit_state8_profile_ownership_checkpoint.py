import numpy as np
import inspect
import pytest

from scripts.early_audit_state8_profile_ownership_checkpoint import (
    load_wrapper,
    checkpoint_step,
    masked_rmse,
    normalize_profile_features,
    observed_future_mask,
    ownership_permutation_null,
)


def test_checkpoint_step_prefers_explicit_intermediate_step():
    assert checkpoint_step({"step": 400, "config": {"diagnostic_max_micro_steps": 200}}) == 400


def test_checkpoint_step_uses_frozen_terminal_diagnostic_contract():
    assert checkpoint_step({"config": {"diagnostic_max_micro_steps": 200}}) == 200


@pytest.mark.parametrize(
    "payload",
    [{}, {"config": {}}, {"config": {"diagnostic_max_micro_steps": None}}, {"step": 0}],
)
def test_checkpoint_step_fails_closed_without_positive_contract(payload):
    with pytest.raises(ValueError):
        checkpoint_step(payload)


def test_checkpoint_audit_uses_the_training_nf4_backbone_precision():
    source = inspect.getsource(load_wrapper)
    assert "load_in_4bit=True" in source
    assert 'bnb_4bit_quant_type="nf4"' in source
    assert "bnb_4bit_use_double_quant=True" in source


def test_masked_rmse_ignores_unsupported_cells():
    prediction = np.asarray([[0.2, 99.0], [0.8, 99.0]])
    target = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    mask = np.asarray([[True, False], [True, False]])
    assert np.isclose(masked_rmse(prediction, target, mask), 0.2)


def test_ownership_permutation_null_detects_student_specific_predictions():
    target = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
    prediction = target.copy()
    mask = np.ones_like(target, dtype=bool)
    result = ownership_permutation_null(
        prediction, target, mask, replicates=200, seed=17
    )
    assert result["native_rmse"] == 0.0
    assert result["delta_permuted_mean_minus_native"] > 0
    assert result["p_one_sided"] < 0.05
    assert result["replicates"] == 200


def test_unsupported_future_rate_placeholders_are_mask_safe():
    prediction = np.asarray([[0.2, 0.5]], dtype=np.float32)
    target = np.asarray([[0.2, 0.5]], dtype=np.float32)
    mask = np.asarray([[True, False]])
    assert masked_rmse(prediction, target, mask) == 0.0


def test_early_audit_layernorm_matches_unit_rms_contract():
    value = np.asarray([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], dtype=np.float32)
    normalized = normalize_profile_features(value, "layernorm")
    np.testing.assert_allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.square(normalized).mean(axis=1), 1.0, atol=2e-5)


def test_seen_scope_uses_only_kcs_named_in_updater_history():
    rows = [{
        "kc_vocabulary": ["a", "b"],
        "state_history": [{"kc_names": ["a"]}],
        "future_eligible_mask": [True, True],
    }]
    np.testing.assert_array_equal(
        observed_future_mask(rows, "seen"), np.asarray([[True, False]])
    )
