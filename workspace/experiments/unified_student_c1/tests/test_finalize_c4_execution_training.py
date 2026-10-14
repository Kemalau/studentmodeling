"""Synthetic lifecycle checks; no remote calls or model measurements."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import finalize_c4_execution_training as finalizer


def test_wait_is_not_completion_or_remote_failure():
    for status in ('watching', 'collecting', 'observation_error'):
        receipt = dict(status=status, pid=123, experiment=finalizer.EXPERIMENT)
        assert not finalizer.ready(receipt, lambda pid: pid == 123)
        with pytest.raises(RuntimeError, match='inspect remote state'):
            finalizer.ready(receipt, lambda pid: False)
    with pytest.raises(RuntimeError, match='needs attention'):
        finalizer.ready(dict(status='needs_attention', experiment=finalizer.EXPERIMENT), lambda p: True)


def test_complete_requires_expected_collector_summary(tmp_path, monkeypatch):
    prefix = tmp_path / 'pilot'
    monkeypatch.setattr(finalizer, 'PREFIX', prefix)
    summary = Path(str(prefix) + '.summary.json')
    receipt = dict(status='complete', experiment=finalizer.EXPERIMENT, summary=str(summary))
    with pytest.raises(FileNotFoundError):
        finalizer.ready(receipt, lambda p: False)
    summary.write_text(json.dumps(dict(status='complete')))
    assert finalizer.ready(receipt, lambda p: False)
    receipt['summary'] = str(tmp_path / 'different.json')
    with pytest.raises(RuntimeError, match='unexpected collector summary'):
        finalizer.ready(receipt, lambda p: True)
