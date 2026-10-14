"""Recover pooled appendix metrics from existing redacted classification scores."""

import hashlib
import json
from pathlib import Path

from build_formal_evaluation_matrix import classification_summary


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    repository = project.parents[2]
    source_root = project / "results/formal_specialist_v1/qwen_amlt"
    matrix_path = project / "results/taskonly_qwen_formal_eval_20260813/formal_specialist_v3/formal_evaluation_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    records = []
    for model_index, model in enumerate(("base", "adapter")):
        sources = list(source_root.rglob(f"{model}_classification/predictions_redacted.jsonl"))
        if len(sources) != 1:
            raise ValueError(f"Expected one {model} classification archive, got {len(sources)}")
        source = sources[0]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for public_task, task, positive in (
            ("T1", "predict_correctness", "correct"),
            ("T5", "predict_error_persistence", "persistent"),
        ):
            metrics = classification_summary(source, task, positive)
            canonical = matrix["rows"][model_index][public_task]
            if metrics["n"] != canonical["n"]:
                raise ValueError(f"Support mismatch: {model}/{public_task}")
            for metric, key in (("accuracy", "value"), ("auroc", "secondary")):
                if abs(metrics[metric] - canonical[key]) > 1e-12:
                    raise ValueError(f"Canonical mismatch: {model}/{public_task}/{metric}")
            records.append({
                "model": matrix["rows"][model_index]["model"],
                "public_task": public_task,
                "task": task,
                "source": str(source.relative_to(repository)),
                "sha256": digest,
                "metrics": metrics,
            })
    output = repository / "paper_iclr/notes/server_classification_scores_20260910.json"
    output.write_text(json.dumps({
        "scope": "Reaggregation of existing label predictions and probabilities; not new generation or semantic judging.",
        "evaluator": "build_formal_evaluation_matrix.classification_summary / formal_metric_utils.binary_metrics",
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: four pooled cells match canonical support, accuracy and AUROC; wrote {output}")


if __name__ == "__main__":
    main()