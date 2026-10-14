#!/usr/bin/env python3
"""Route the current item through the decoder query.

The recurrent state is a summary of observed student interactions.  The item
being diagnosed is also the current query and therefore its non-student item
fields must remain visible to the state-only decoder.  The previous manifest
left the decoder with only a generic instruction, forcing State8 to reconstruct
the entire question as well as the student-specific response and misconception.

This transform keeps the information bottleneck intact:

    observed interactions -> recurrent state -> decoder
    current item --------------------------> decoder query

Raw history and student responses are never copied into the decode prompt.
Canonical labels, targets, candidate taxonomy entries, and correctness are
also rejected from the current-query payload so the transform cannot leak
supervision.  The full current observation remains in the updater: the state,
not the query, must carry what the student actually answered.
"""

from __future__ import annotations

import json
import re
from typing import Any


OBSERVATION = re.compile(
    r"<observed_interaction>(.*?)</observed_interaction>", re.I | re.S
)
QUERY_OPEN = "<query>"
FORBIDDEN_KEYS = {
    "target",
    "target_label",
    "misconception",
    "misconception_id",
    "misconception_label",
    "diagnosis",
    "evidence",
    "candidate_labels",
    "candidate_targets",
}
QUERY_VISIBLE_KEYS = (
    "kc_ids",
    "kc_names",
    "question_text",
    "problem_text",
    "options",
    "correct_response",
    "correct_response_id",
    "correct_answer",
    "answer_type",
    "question_type",
    "problem_part",
    "problem_set_id",
)


def query_visible_value(key: str, value: Any) -> Any:
    """Remove answer-like student observations embedded in source text."""
    if key not in {"question_text", "problem_text"} or not isinstance(value, str):
        return value
    # Eedi stores the wrong response inside ``question_text`` as a legacy
    # ``### Incorrect:`` suffix.  Keep the item and its known correct answer,
    # but require the recurrent state to carry what this student answered.
    return re.sub(r"\n###\s*Incorrect\s*:.*\Z", "", value, flags=re.I | re.S).rstrip()


def observed_payload(update_text: str) -> dict[str, Any]:
    """Parse one audited updater view and reject target-bearing fields."""
    matches = OBSERVATION.findall(str(update_text))
    if len(matches) != 1:
        raise ValueError("Current update must contain one observed_interaction")
    value = json.loads(matches[0])
    if not isinstance(value, dict) or not value:
        raise ValueError("Current observed interaction must be a non-empty object")
    leaked = FORBIDDEN_KEYS.intersection(value)
    if leaked:
        raise ValueError(f"Current-query observation contains target fields: {sorted(leaked)}")
    return value


def inject_current_query(text: str, observation: dict[str, Any]) -> str:
    """Add only non-student current-item fields to the query segment."""
    value = str(text)
    if value.count(QUERY_OPEN) != 1 or value.count("</query>") != 1:
        raise ValueError("Decode prompt must contain exactly one query segment")
    if "<current_item>" in value:
        raise ValueError("Decode prompt already contains a current item")
    item = {
        key: query_visible_value(key, observation[key])
        for key in QUERY_VISIBLE_KEYS
        if key in observation and observation[key] is not None
    }
    if not item:
        raise ValueError("Current observation contains no query-visible item fields")
    payload = json.dumps(
        item,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    routed = (
        f"{QUERY_OPEN}\n"
        f"<current_item>{payload}</current_item>\n"
        "Use the current item together with the student state. The state carries "
        "the student's observed responses; do not infer a response from the item alone.\n"
    )
    return value.replace(QUERY_OPEN, routed, 1)


def apply_current_query_routing(
    rows: list[dict[str, Any]], *, sources: set[str] | None = None
) -> dict[str, int]:
    """Expose current item context while keeping all responses in recurrence.

    The full-history teacher receives the same current-item query as the
    student, but can additionally inspect ``direct_text``'s history.  The
    student ``decode_text`` never receives that history or the student's
    response.  Keeping the query identical makes answer-logit distillation a
    comparison of information routes rather than a comparison of prompts.
    """
    changed = 0
    teacher_queries = 0
    state_updates = 0
    by_source: dict[str, int] = {}
    for row in rows:
        if str(row.get("task") or "") != "diagnose_misconception":
            continue
        source = str(row.get("source") or "unknown")
        if sources is not None and source not in sources:
            continue
        updates = [str(value) for value in (row.get("update_texts") or [])]
        if not updates:
            raise ValueError(
                f"Diagnosis row lacks an audited current update: {row.get('example_id')}"
            )
        observation = observed_payload(updates[-1])
        row["decode_text"] = inject_current_query(
            str(row.get("decode_text") or row.get("prompt") or ""), observation
        )
        if "direct_text" in row:
            row["direct_text"] = inject_current_query(
                str(row["direct_text"]), observation
            )
            teacher_queries += 1
        row["current_query_routing"] = "current_item_visible_response_state_only_v1"
        row["raw_prior_history_visible_to_decoder"] = False
        row["current_item_visible_to_decoder"] = True
        row["student_response_visible_to_decoder"] = False
        changed += 1
        state_updates += len(updates)
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "current_query_routed_rows": changed,
        "current_query_teacher_rows": teacher_queries,
        "current_query_state_updates": state_updates,
        **{
            f"current_query_source_{source}": count
            for source, count in sorted(by_source.items())
        },
    }
