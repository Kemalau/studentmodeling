from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def test_publisher_verifies_and_commits_receipt_last() -> None:
    text=(ROOT/'scripts/watch_state8_v5_immutable_publish.sh').read_text()
    assert 'remote_passed' in text
    assert "c['micro_steps']==1000" in text
    assert "[64,8,5120]" in text
    assert text.index('metrics/completion.json') < text.rindex('metrics/immutable_receipt.json')
    assert 'immutable_published' in text
    assert 'amlt cancel' not in text
    assert "model/adapter_config.json" in text
    assert "--include 'adapter_config.json'" in text
    # AzCopy include patterns are filename filters, not relative paths.
    include_args = [
        part.split("'", 2)[1]
        for part in text.split('--include ')[1:]
        if part.startswith("'")
    ]
    assert include_args
    assert all('/' not in pattern for pattern in include_args)
