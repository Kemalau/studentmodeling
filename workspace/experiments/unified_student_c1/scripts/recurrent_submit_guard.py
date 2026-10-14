#!/usr/bin/env python3
"""Pure fail-closed classifiers for the recurrent M2 submission controller."""

from __future__ import annotations

import re


MISSING_EXPERIMENT = re.compile(
    r"not found|cannot find|does not exist|no experiment|unknown experiment", re.I
)
ACTIVE_JOB_STATUS = re.compile(r"running|queued|preparing|paused|resuming", re.I)
FORWARD_PROGRESS_JOB_STATUS = re.compile(r"running|queued|preparing|resuming", re.I)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
JOB_LINE = re.compile(r"^\s*(?:(?::?\d+)\s+)?(?::\S+)\s+(?P<rest>.+)$")
KNOWN_JOB_STATUS = re.compile(
    r"\b(preparing|queued|running|paused|resuming|pass|failed|killed|cancelled)\b",
    re.I,
)


def classify_experiment_status(returncode: int, output: str) -> str:
    if returncode == 0:
        return "exists"
    if MISSING_EXPERIMENT.search(output):
        return "missing"
    return "ambiguous"


def has_active_job(output: str) -> bool:
    """Return true for scheduler states that may make forward progress."""
    return ACTIVE_JOB_STATUS.search(output) is not None


def has_forward_progress_job(output: str) -> bool:
    """Return true only for scheduler states expected to advance unaided."""
    return FORWARD_PROGRESS_JOB_STATUS.search(output) is not None


def all_expected_jobs_pass(output: str, expected_jobs: int) -> bool:
    """Recognize both AMLT aggregate summaries and single-job status tables."""
    if expected_jobs <= 0:
        raise ValueError("expected_jobs must be positive")
    text = ANSI_ESCAPE.sub("", output)
    aggregate = re.search(
        rf"\b{expected_jobs}\s+jobs?\s+(?:—|-)\s+{expected_jobs}\s+pass\b",
        text,
        re.I,
    )
    if aggregate:
        return True
    statuses: list[str] = []
    for line in text.splitlines():
        match = JOB_LINE.match(line)
        if not match:
            continue
        state = KNOWN_JOB_STATUS.search(match.group("rest"))
        if state:
            statuses.append(state.group(1).lower())
    return len(statuses) == expected_jobs and all(state == "pass" for state in statuses)


def terminal_nonpass_job_names(output: str) -> list[str]:
    """Return failed/killed/cancelled job names from an AMLT status table."""
    text = ANSI_ESCAPE.sub("", output)
    names: list[str] = []
    for line in text.splitlines():
        match = JOB_LINE.match(line)
        if not match:
            continue
        state = KNOWN_JOB_STATUS.search(match.group("rest"))
        if state and state.group(1).lower() in {"failed", "killed", "cancelled"}:
            prefix = line[: match.start("rest")].strip().split()
            candidates = [value for value in prefix if value.startswith(":")]
            if candidates:
                name = candidates[-1].lstrip(":")
                if name:
                    names.append(name)
    return names


def paused_job_names(output: str) -> list[str]:
    """Return paused job names so a controller can recover them explicitly."""
    text = ANSI_ESCAPE.sub("", output)
    names: list[str] = []
    for line in text.splitlines():
        match = JOB_LINE.match(line)
        if not match:
            continue
        state = KNOWN_JOB_STATUS.search(match.group("rest"))
        if state and state.group(1).lower() == "paused":
            prefix = line[: match.start("rest")].strip().split()
            candidates = [value for value in prefix if value.startswith(":")]
            if candidates:
                name = candidates[-1].lstrip(":")
                if name:
                    names.append(name)
    return names


def passed_job_names(output: str) -> list[str]:
    """Return explicitly passed job names from an AMLT status table.

    This is intentionally row based: an aggregate ``N jobs — N pass`` line
    proves experiment success but cannot identify which shards may be reused
    across a targeted recovery experiment.
    """
    text = ANSI_ESCAPE.sub("", output)
    names: list[str] = []
    for line in text.splitlines():
        match = JOB_LINE.match(line)
        if not match:
            continue
        state = KNOWN_JOB_STATUS.search(match.group("rest"))
        if state and state.group(1).lower() == "pass":
            prefix = line[: match.start("rest")].strip().split()
            candidates = [value for value in prefix if value.startswith(":")]
            if candidates:
                name = candidates[-1].lstrip(":")
                if name:
                    names.append(name)
    return names
