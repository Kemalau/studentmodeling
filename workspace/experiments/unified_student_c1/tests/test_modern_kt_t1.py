import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_modern_kt_t1_splits import EXPECTED, EXPECTED_SOURCES  # noqa: E402
from train_modern_kt_t1 import T1Dataset, import_model_class, init_model, maps, model_probabilities  # noqa: E402


def test_frozen_manifest_has_five_sources_and_no_overlap() -> None:
    value = json.loads((ROOT / "data/processed/modern_kt_t1_five_source_v1/manifest.json").read_text())
    assert {split: value["splits"][split]["rows"] for split in EXPECTED} == EXPECTED
    assert all(set(value["splits"][split]["by_source"]) == EXPECTED_SOURCES for split in EXPECTED)
    assert not any(value["overlap"].values())


def test_dataset_never_places_target_label_in_model_response_input() -> None:
    base = {
        "dataset": "toy",
        "metadata": {"mix_source": "toy", "question_id": "q2", "skill_id": "c2"},
        "query": {"question_id": "q2", "kc_ids": ["c2"]},
        "history": [{"question_id": "q1", "kc_ids": ["c1"], "correctness": 1}],
        "target_label": "correct",
    }
    skill_map, question_map = maps([base])
    correct = T1Dataset([base], skill_map, question_map, 4)[0]
    wrong_row = dict(base, target_label="incorrect")
    wrong = T1Dataset([wrong_row], skill_map, question_map, 4)[0]
    assert torch.equal(correct["responses"], wrong["responses"])
    assert correct["responses"][correct["target_position"]].item() == 0
    assert correct["responses"][-1].item() == -1
    assert correct["label"].item() == 1
    assert wrong["label"].item() == 0


def test_modern_kt_target_probability_is_invariant_to_dummy_target_response() -> None:
    class Args:
        d_model = 128
        dropout = 0.1
        max_seq_len = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {
        "skills": torch.tensor([[1, 2, 3, 0]], device=device),
        "questions": torch.tensor([[1, 2, 3, 0]], device=device),
        "responses": torch.tensor([[1, 0, 0, -1]], device=device),
        "target_position": torch.tensor([2], device=device),
    }
    changed = {key: value.clone() for key, value in batch.items()}
    changed["responses"][0, 2] = 1
    for model_name in ("akt", "simplekt", "dtransformer", "rekt", "ukt"):
        model_class = import_model_class(ROOT / "third_party/pykt_toolkit", model_name)
        model = init_model(model_name, model_class, n_skills=8, n_questions=8, args=Args()).to(device)
        model.eval()
        with torch.inference_mode():
            first = model_probabilities(model_name, model, batch)
            second = model_probabilities(model_name, model, changed)
        assert torch.equal(first, second), model_name
