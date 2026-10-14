from types import SimpleNamespace

from scripts.train_state8_repaired_teacher_curriculum import repaired_dual_path_weights


def test_repaired_curriculum_keeps_endpoint_kd_on():
    args = SimpleNamespace(history_dropout_curriculum=True, dual_path_curriculum=True)
    assert [repaired_dual_path_weights(args, value) for value in (0, .2, .4, .6, .8)] == [
        (1.0, 0.0, 0.0),
        (0.75, 0.25, 1.0),
        (0.5, 0.5, 1.0),
        (0.25, 0.75, 1.0),
        (0.0, 1.0, 1.0),
    ]
