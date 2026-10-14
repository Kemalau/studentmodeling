"""Shared validation helpers for source and AMLT-normalized run configs."""

from __future__ import annotations

import subprocess
from typing import Any


def command_text(value: Any) -> str:
    """Normalize source and AMLT-rendered command representations.

    Source configs store ``command`` as a scalar string while AMLT 11.14.2
    renders it as a one-element list.  Calling ``str(list)`` escapes quotes and
    newlines, which makes exact fail-closed checks reject an otherwise identical
    rendered command.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def validate_shell_syntax(command: str, job_name: str) -> None:
    """Validate the shell after AMLT literal-dollar template expansion."""
    shell_command = command.replace("$$", "$")
    result = subprocess.run(
        ["bash", "-n"],
        input=shell_command,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"Shell syntax error in {job_name}: {detail}")


def target_matches(config_target: Any, workspace: str) -> bool:
    """Accept only the two equivalent target encodings AMLT 11.14.2 emits.

    Source templates use ``service: manifold`` and an ``NAME@RESOURCE_GROUP``
    workspace reference. ``amlt run --dump`` removes the service and splits the
    workspace into name/resource-group fields. No other target is accepted.
    """
    if not isinstance(config_target, dict) or "@" not in workspace:
        return False
    workspace_name, resource_group = workspace.split("@", 1)
    source = {
        "service": "manifold",
        "name": "Feeds",
        "workspace_name": workspace,
    }
    normalized = {
        "name": "Feeds",
        "workspace_name": workspace_name,
        "workspace_resource_group": resource_group,
    }
    normalized_with_service = {
        "service": "sing",
        **normalized,
    }
    return config_target in (source, normalized, normalized_with_service)
