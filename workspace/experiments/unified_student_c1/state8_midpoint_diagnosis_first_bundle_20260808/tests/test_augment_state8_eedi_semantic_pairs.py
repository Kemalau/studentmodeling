from scripts.augment_state8_eedi_semantic_pairs import augment


def row(example_id: str, label: str, candidates: list[str]):
    return {
        "example_id": example_id,
        "task": "diagnose_misconception",
        "source": "eedi_mining",
        "target_label": label,
        "candidate_labels": candidates,
        "update_texts": [f"update {example_id}"],
    }


def test_pair_augmentation_uses_same_label_and_hard_negative() -> None:
    corpus = [
        row("a1", "A", ["A", "B"]),
        row("a2", "A", ["A", "B"]),
        row("b1", "B", ["A", "B"]),
    ]
    rows, receipt = augment([corpus[0]], corpus, seed=7)
    value = rows[0]
    assert value["eedi_positive_label"] == "A"
    assert value["eedi_negative_label"] == "B"
    assert value["eedi_positive_id"] == "a2"
    assert value["eedi_pair_supervision_visibility"] == "train_only_not_decoder_input"
    assert receipt["pair_coverage"] == 1.0
    assert receipt["canonical_labels_visible_to_decoder"] is False


def test_singleton_label_is_left_unpaired() -> None:
    corpus = [
        row("a1", "A", ["A", "B"]),
        row("b1", "B", ["A", "B"]),
        row("b2", "B", ["A", "B"]),
    ]
    rows, receipt = augment([corpus[0], corpus[1]], corpus, seed=7)
    assert "eedi_semantic_pair" not in rows[0]
    assert receipt["missing_positive"] == 1
