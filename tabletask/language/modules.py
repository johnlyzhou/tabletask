"""Vocabulary modules a task imports with `uses`, and the observations they need."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import UnsupportedError
from .transformations import inline


def load_modules(uses: dict, root: Path) -> tuple[dict, dict]:
    """The words and observations each module in `uses` provides, read from its file."""
    words, observations = {}, {}
    for namespace, ref in uses.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", namespace):
            raise ValueError(f"invalid module namespace {namespace!r}")
        if not isinstance(ref, dict) or set(ref) != {"path", "version"}:
            raise ValueError(f"module {namespace}: requires path and exact version")
        module = json.loads((root / ref["path"]).read_text())
        if not isinstance(module, dict):
            raise ValueError(f"module {namespace}: expected an object")
        if any(not isinstance(module.get(key, {}), dict) for key in ("observations", "words")):
            raise ValueError(f"module {namespace}: observations and words must be objects")
        if set(module) - {"module", "version", "observations", "words"}:
            raise ValueError(f"module {namespace}: unknown fields")
        if module.get("module") != namespace or module.get("version") != ref["version"]:
            raise ValueError(f"module {namespace}: name or version mismatch")
        if not isinstance(ref["version"], str) or not re.fullmatch(
            r"\d+\.\d+(?:\.\d+)?", ref["version"]
        ):
            raise ValueError(f"module {namespace}: version must be an exact numeric version")
        for name, definition in module.get("observations", {}).items():
            if not isinstance(definition, dict) or set(definition) != {
                "of",
                "kind",
                "unit",
                "description",
            }:
                raise ValueError(
                    f"{namespace}/{name}: observation requires of, kind, unit, description"
                )
            if not isinstance(definition["of"], list) or any(
                not isinstance(p, str) or not p for p in definition["of"]
            ):
                raise ValueError(f"{namespace}/{name}: observation parameters must be entity names")
            if not isinstance(definition["description"], str) or not definition["description"]:
                raise ValueError(f"{namespace}/{name}: an observation description is required")
            if definition["kind"] not in ("number", "boolean"):
                raise ValueError(f"{namespace}/{name}: kind must be number or boolean")
            if not isinstance(definition["unit"], str) or not definition["unit"]:
                raise ValueError(
                    f"{namespace}/{name}: an explicit unit is required (1 for dimensionless)"
                )
            capability = f"{namespace}@{ref['version']}/{name}"
            observations[capability] = definition
            op = "reading" if definition["kind"] == "number" else "observed"
            words[f"{namespace}/{name}"] = {
                "of": definition["of"],
                "is": {op: [capability, *definition["of"]]},
            }
        for name, definition in module.get("words", {}).items():
            key = f"{namespace}/{name}"
            if key in words:
                raise ValueError(f"duplicate module word {key}")
            words[key] = definition
    return words, observations


def requirements(task) -> frozenset[str]:
    """The observations a task needs, in its start, goal and the words it defines."""
    task = inline(task)
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for name, args in node.items():
                if name in ("reading", "observed"):
                    found.add(args[0])
                walk(args)
        elif isinstance(node, list):
            for sub in node:
                walk(sub)

    for stage in task.goal.stages.values():
        walk(stage.condition)
    for condition in task.goal.always.values():
        walk(condition)
    walk(task.scene.start)
    return frozenset(found)


def require_capabilities(task, capabilities, operation="evaluation"):
    """Raise UnsupportedError if the task needs an observation not in `capabilities`."""
    missing = requirements(task) - set(capabilities)
    if missing:
        raise UnsupportedError(operation, missing)
