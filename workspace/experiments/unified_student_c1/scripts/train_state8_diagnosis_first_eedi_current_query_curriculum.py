#!/usr/bin/env python3
"""State8 Eedi curriculum with the current item routed beside the state.

This is the matched intervention for the generic-query curriculum.  Only
non-student Eedi item fields are copied into the teacher/student decode query;
the observed wrong response remains exclusively in the recurrent updater and
the final decoder remains history-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import train_state8_diagnosis_first_eedi_curriculum as base
import train_state8_taxonomy_teacher_curriculum as taxonomy
from state8_current_query_routing import apply_current_query_routing


ROUTED_SOURCES = {"eedi_mining"}


def main() -> None:
    output_dir = base.cli_path("--output-dir")
    original_parse = argparse.ArgumentParser.parse_args
    original_view = taxonomy.apply_state8_natural_curriculum_view
    audits: list[dict[str, Any]] = []

    def parse(parser, *args, **kwargs):
        namespace = original_parse(parser, *args, **kwargs)
        namespace.current_query_routing = (
            "current_item_visible_response_state_only_v1:eedi_only"
        )
        namespace.current_query_sources = sorted(ROUTED_SOURCES)
        namespace.current_item_visible_to_decoder = True
        namespace.student_response_visible_to_decoder = False
        namespace.raw_history_visible_to_decoder = False
        namespace.current_query_curriculum_entrypoint_sha256 = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        return namespace

    def routed_view(rows):
        audit = dict(original_view(rows))
        routing = apply_current_query_routing(rows, sources=ROUTED_SOURCES)
        audit.update(routing)
        if int(routing["current_query_routed_rows"]) > 0:
            audits.append(routing)
        return audit

    argparse.ArgumentParser.parse_args = parse
    taxonomy.apply_state8_natural_curriculum_view = routed_view
    try:
        base.main()
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", "0")) != 0:
            return
        config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
        routed_rows = sum(int(row["current_query_routed_rows"]) for row in audits)
        teacher_rows = sum(int(row["current_query_teacher_rows"]) for row in audits)
        source_rows = sum(
            int(row.get("current_query_source_eedi_mining", 0)) for row in audits
        )
        if routed_rows <= 0 or teacher_rows != routed_rows or source_rows != routed_rows:
            raise RuntimeError(
                "Current-query curriculum did not align all routed teacher/student rows"
            )
        expected = "current_item_visible_response_state_only_v1:eedi_only"
        if config.get("current_query_routing") != expected:
            raise RuntimeError("Current-query route was not serialized")
        if config.get("student_response_visible_to_decoder") is not False:
            raise RuntimeError("Student response leaked to the decoder")
        receipt = {
            "stage": "state8_diagnosis_first_eedi_current_query_curriculum_v1",
            "status": "complete",
            "state_token_count": 8,
            "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
            "current_query_routing": expected,
            "current_query_sources": sorted(ROUTED_SOURCES),
            "current_query_routed_rows_across_reads": routed_rows,
            "current_query_teacher_rows_across_reads": teacher_rows,
            "current_item_visible_to_decoder": True,
            "student_response_visible_to_decoder": False,
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "final_inference_route": "strict_state_only_plus_current_item",
            "same_question_causal_replay_required": True,
            "semantic_judge_authoritative": True,
            "exact_match_authoritative": False,
            "canonical_label_role": "auxiliary_supervision_and_reporting",
            "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "audits": audits,
        }
        (output_dir / "eedi_current_query_curriculum.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        taxonomy.apply_state8_natural_curriculum_view = original_view


if __name__ == "__main__":
    main()
