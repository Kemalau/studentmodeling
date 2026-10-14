#!/usr/bin/env python3
"""State8 semantic-causal repair with an explicit taxonomy-label contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import train_recurrent_student_m4 as trainer
import train_state8_semantic_causal_repair as semantic
from state8_natural_curriculum_view import apply_state8_natural_curriculum_view
from state8_taxonomy_contract import apply_state8_taxonomy_contract


def main() -> None:
    original_read = trainer.read_jsonl
    original_apply = semantic.repair.apply_state8_natural_curriculum_view
    original_parse = argparse.ArgumentParser.parse_args
    audits: list[dict[str, object]] = []
    output_dir: Path | None = None

    def combined_natural_view(rows):
        natural = apply_state8_natural_curriculum_view(rows)
        taxonomy = apply_state8_taxonomy_contract(rows)
        audits.append({"scope": "real_rows", **natural, **taxonomy})
        return {**natural, **taxonomy}

    def read(path):
        rows = original_read(path)
        audit = apply_state8_taxonomy_contract(rows)
        if int(audit["taxonomy_contract_rows"]):
            audits.append({"scope": "loaded_rows", "path": str(path), **audit})
        return rows

    def parse(parser, *args, **kwargs):
        nonlocal output_dir
        namespace = original_parse(parser, *args, **kwargs)
        output_dir = Path(namespace.output_dir)
        namespace.taxonomy_label_contract = "source_aware_no_answer_labels_v1"
        namespace.taxonomy_candidates_visible_to_decoder = False
        return namespace

    trainer.read_jsonl = read
    semantic.repair.apply_state8_natural_curriculum_view = combined_natural_view
    argparse.ArgumentParser.parse_args = parse
    try:
        semantic.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) == 0:
            if output_dir is None:
                raise RuntimeError("Taxonomy repair did not observe the output directory")
            total = sum(int(row.get("taxonomy_contract_rows", 0)) for row in audits)
            if total <= 0:
                raise RuntimeError("Taxonomy repair applied no diagnosis contracts")
            receipt = {
                "stage": "state8_taxonomy_causal_repair_v1",
                "status": "complete",
                "contract": "source_aware_no_answer_labels_v1",
                "candidate_taxonomy_visible_to_decoder": False,
                "raw_history_visible_to_decoder": False,
                "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "audits": audits,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "taxonomy_causal_repair.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        trainer.read_jsonl = original_read
        semantic.repair.apply_state8_natural_curriculum_view = original_apply
        argparse.ArgumentParser.parse_args = original_parse


if __name__ == "__main__":
    main()

