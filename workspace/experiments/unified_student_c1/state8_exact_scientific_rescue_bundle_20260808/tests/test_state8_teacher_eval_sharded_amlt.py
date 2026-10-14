from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_teacher_eval_sharded_amlt import load, validate  # noqa: E402


def test_teacher_eval_sharded_config_passes() -> None:
    validate(load(ROOT / "amlt/feeds_state8_teacher_eval_sharded_20260805.yaml"))
    text = (ROOT / "amlt/feeds_state8_teacher_eval_sharded_20260805.yaml").read_text()
    assert "state8_teacher_clean_prompt_v2" in text
    assert "rewrite_state8_teacher_prompts.py" in text
    assert "78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5" in text


def test_teacher_eval_recovery_subset_passes() -> None:
    config = load(ROOT / "amlt/feeds_state8_teacher_eval_sharded_20260805.yaml")
    selected = {"state8-full-history-teacher-eval-shard2"}
    config["jobs"] = [job for job in config["jobs"] if job["name"] in selected]
    validate(config, selected)


def test_teacher_eval_16way_config_passes() -> None:
    path = ROOT / "amlt/feeds_state8_teacher_eval_16way_20260805.yaml"
    validate(load(path), num_shards=16)
    text = path.read_text()
    assert '--num-shards 16' in text
    assert "state8-full-history-teacher-eval-shard15" in text
