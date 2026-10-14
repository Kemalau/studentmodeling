from scripts import snapshot_state8_checkpoint_blob as snapshot
import json


def test_snapshot_copies_only_required_checkpoint_files_receipt_last():
    assert snapshot.FILES == (
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "recurrent_state.pt",
    )


def test_large_snapshot_uses_async_copy_and_waits_for_success(monkeypatch):
    calls = []
    statuses = iter([("pending", ""), ("success", "")])

    def fake_az(*arguments):
        calls.append(arguments)
        return '{"copy_status": "pending"}'

    monkeypatch.setattr(snapshot, "az", fake_az)
    monkeypatch.setattr(snapshot, "copy_status", lambda _: next(statuses))
    monkeypatch.setattr(snapshot.time, "sleep", lambda _: None)

    snapshot.copy("source/model.safetensors", "destination/model.safetensors")

    assert "--requires-sync" not in calls[0]
    assert calls[0][:2] == ("copy", "start")


def test_snapshot_copy_rejects_failed_async_status(monkeypatch):
    monkeypatch.setattr(snapshot, "az", lambda *args: '{"copy_status": "pending"}')
    monkeypatch.setattr(snapshot, "copy_status", lambda _: ("failed", "source denied"))

    try:
        snapshot.copy("source", "destination")
    except RuntimeError as error:
        assert "source denied" in str(error)
    else:
        raise AssertionError("failed Blob copy was accepted")


def test_snapshot_accepts_distributed_round_receipt(tmp_path, monkeypatch):
    receipt = tmp_path / "checkpoint_completion.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "round": 1000,
                "sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64},
            }
        ),
        encoding="utf-8",
    )
    copied = []
    monkeypatch.setattr(
        snapshot,
        "exists",
        lambda name: name != "destination/checkpoint_completion.json",
    )
    monkeypatch.setattr(snapshot, "copy", lambda source, destination: copied.append((source, destination)))

    result = snapshot.snapshot("source", "destination", receipt)

    assert result["step"] == 1000
    assert copied[-1] == (
        "source/checkpoint_completion.json",
        "destination/checkpoint_completion.json",
    )
