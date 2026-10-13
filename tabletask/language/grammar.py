"""Checks on conditions, and listings of the words they are written in."""

from __future__ import annotations

import inspect
from difflib import get_close_matches
from typing import Any

import msgspec

from . import evaluation, kernel
from .schema import ROTATIONS, Task
from .vocabulary import (
    CONNECTIVES,
    PLACEHOLDER,
    QUANTIFIERS,
    TEMPORAL,
    TIME_WORDS,
    VOCABULARY,
)


def check_task(task: Task) -> Task:
    """Raise ValueError if the task's conditions, stages or variables are malformed."""
    for name, obj in task.scene.objects.items():
        rotation = obj.sample.rotation
        if isinstance(rotation, str) and rotation not in ROTATIONS:
            raise ValueError(
                f"object {name!r}: rotation {rotation!r} is not one of {sorted(ROTATIONS)}"
            )
    states = _check_states(task)
    for name, stage in task.goal.stages.items():
        for prerequisite in stage.after:
            if prerequisite not in task.goal.stages:
                raise ValueError(f"stage {name!r}: after unknown stage {prerequisite!r}")
        check_condition(stage.condition, f"stage {name!r}", states)
    for name, tree in task.goal.always.items():
        check_condition(tree, f"{name}", states)
    if task.scene.start is not None:
        check_condition(task.scene.start, "start", states)
    task.goal.order()
    for text in _strings(msgspec.to_builtins(task, enc_hook=str)):
        for name in PLACEHOLDER.findall(text):
            if name not in task.variables:
                raise ValueError(f"{task.id}: {{{name}}} is used but not declared in variables")
    return task


def _check_states(task: Task) -> dict[str, dict]:
    """Check the words a task defines for itself, and that none defines itself."""
    reserved = (
        set(VOCABULARY)
        | set(kernel.RELATIONS)
        | set(kernel.TERMS)
        | set(QUANTIFIERS)
        | set(CONNECTIVES)
        | set(TEMPORAL)
    )
    for name, definition in task.states.items():
        if name in reserved:
            raise ValueError(f"{task.id}: state {name!r} would shadow a word of the language")
        if not isinstance(definition, dict) or set(definition) != {"of", "is"}:
            raise ValueError(f"{task.id}: state {name!r} must be {{of: [parameters], is: tree}}")
        if not isinstance(definition["of"], list) or not definition["of"]:
            raise ValueError(f"{task.id}: state {name!r} needs at least one parameter")
    for name, definition in task.states.items():
        parameters = [
            p if isinstance(p, str) else p[0] if isinstance(p, list) and len(p) == 2 else None
            for p in definition["of"]
        ]
        if any(not isinstance(p, str) or not p for p in parameters) or len(set(parameters)) != len(
            parameters
        ):
            raise ValueError(f"state {name!r}: parameters must have unique names")
        check_condition(definition["is"], f"state {name!r}", task.states)

    def uses(node) -> set[str]:
        if isinstance(node, dict):
            return set(node) & set(task.states) | set().union(*(uses(v) for v in node.values()))
        if isinstance(node, list):
            return set().union(*(uses(v) for v in node))
        return set()

    checked = set()

    def visit(name, trail=()):
        if name in trail:
            raise ValueError("recursive vocabulary definition: " + " -> ".join((*trail, name)))
        if name not in checked:
            for other in uses(task.states[name]["is"]):
                visit(other, (*trail, name))
            checked.add(name)

    for name in task.states:
        visit(name)
    return task.states


def check_condition(tree: Any, where: str, local: dict | None = None) -> None:
    """Raise ValueError if a compiled condition uses an unknown word or wrong arguments."""
    if not isinstance(tree, dict) or len(tree) != 1:
        raise ValueError(f"{where}: each node must be one {{word: arguments}} pair, got {tree!r}")
    name, args = evaluation.unpack_expression(tree)
    words = {**VOCABULARY, **(local or {})}

    if name in QUANTIFIERS:
        if not 2 <= len(args) <= 4:
            raise ValueError(
                f"{where}: {name!r} takes a group, a condition, and optional binder names"
            )
        group = args[0]
        if not (isinstance(group, list) or (isinstance(group, str) and group[:1] in "#{")):
            raise ValueError(
                f"{where}: {name!r} ranges over a list of names or a #tag, not {group!r}"
            )
        check_condition(args[1], where, local)
        return
    if name in TIME_WORDS:
        expected = 2 if name == "times" else 1
        if len(args) != expected:
            raise ValueError(f"{where}: {name} takes {expected} arguments")
        if name == "times" and (type(args[1]) is not int or args[1] < 1):
            raise ValueError(f"{where}: times needs a positive integer count")
        check_condition(args[0], where, local)
        return
    if name in CONNECTIVES:
        if name == "not" and len(args) != 1:
            raise ValueError(f"{where}: not takes one condition")
        for sub in args:
            check_condition(sub, where, local)
        return
    if name in words:
        needed = sum(1 for parameter in words[name]["of"] if isinstance(parameter, str))
        total = len(words[name]["of"])
        if not needed <= len(args) <= total:
            count = str(needed) if needed == total else f"{needed} to {total}"
            raise ValueError(f"{where}: {name!r} takes {count} arguments, got {len(args)}")
    elif (
        name not in kernel.ARITHMETIC and name not in kernel.RELATIONS and name not in kernel.TERMS
    ):
        near = get_close_matches(name, [*words, *kernel.RELATIONS, *kernel.TERMS], 3)
        hint = f"; did you mean {', '.join(repr(n) for n in near)}?" if near else ""
        raise ValueError(f"{where}: unknown word {name!r}{hint}")
    if name in kernel.ARITHMETIC and len(args) != 2:
        raise ValueError(f"{where}: {name} takes two arguments")
    function = kernel.RELATIONS.get(name) or kernel.TERMS.get(name)
    if function:
        try:
            inspect.signature(function).bind(None, *args)
        except TypeError as error:
            raise ValueError(f"{where}: {name}: {error}") from error

    def nested(arg):
        if isinstance(arg, dict):
            check_condition(arg, where, local)
        elif isinstance(arg, list):
            for item in arg:
                nested(item)

    for arg in args:
        nested(arg)


def names(task: Task) -> list[tuple[str, str]]:
    """Every scene name a task refers to, with where it is referred to."""
    found = [(obj.sample.rel, f"object {n!r} sample") for n, obj in task.scene.objects.items()]
    for stage_name, stage in task.goal.stages.items():
        found += [(n, f"stage {stage_name!r}") for n in names_in(stage.condition)]
    for name, tree in task.goal.always.items():
        found += [(n, name) for n in names_in(tree)]
    if task.scene.start is not None:
        found += [(n, "start") for n in names_in(task.scene.start)]
    return found


def names_in(tree: Any) -> list[str]:
    """Every object or region a condition refers to, in order."""
    if isinstance(tree, list):
        return [n for item in tree for n in names_in(item)]
    if not isinstance(tree, dict):
        return []
    name, args = evaluation.unpack_expression(tree)
    if name in ("reading", "observed"):
        return [a for a in args[1:] if isinstance(a, str)]
    if name in QUANTIFIERS:
        return []
    if name in TIME_WORDS:
        return names_in(args[0])
    if name in CONNECTIVES or name in kernel.ARITHMETIC:
        return [n for sub in args for n in names_in(sub)]

    function = kernel.RELATIONS.get(name) or kernel.TERMS.get(name)
    defaults = (
        [p.default for p in inspect.signature(function).parameters.values() if p.name != "s"]
        if function is not None
        else []
    )
    found = []
    for i, arg in enumerate(args):
        if i < len(defaults) and isinstance(defaults[i], str):
            continue  # an option such as an axis name, not an object or region
        if isinstance(arg, (dict, list)):
            found += names_in(arg)
        elif isinstance(arg, str):
            found.append(arg)
    return found


def words() -> str:
    """A reference listing of every word a condition may use, and its arguments."""
    lines = ["defined words"]
    for name, definition in VOCABULARY.items():
        parameters = [p if isinstance(p, str) else f"[{p[0]}={p[1]}]" for p in definition["of"]]
        body = msgspec.json.encode(definition["is"]).decode()
        lines.append(f"  {name:14s} {' '.join(parameters):34s} := {body}")
    lines.append("relations")
    for name, function in kernel.RELATIONS.items():
        lines.append(f"  {name:14s} {_signature(function):34s} {_summary(function)}")
    lines.append("terms (measurements, giving a number)")
    for name, function in kernel.TERMS.items():
        lines.append(f"  {name:14s} {_signature(function):34s} {_summary(function)}")
    lines.append("a condition is [word, argument, ...]; a list of conditions means all of them")
    lines.append(
        "quantifiers: every some no every_pair each, over [names] or #tag, binding it (and other)"
    )
    lines.append("logic: and or not   order: then (as the whole goal)   start only: fixed")
    lines.append(
        "time: once (has been true) times (how many times it became true) initially (at the start)"
    )
    lines.append(
        'counting: ["at_least", ["tally", [conditions]], n]   arithmetic on numbers: + - *'
    )
    return "\n".join(lines)


def _summary(function) -> str:
    """The first line of a function's docstring."""
    lines = (function.__doc__ or "").strip().splitlines()
    return lines[0] if lines else ""


def _signature(function) -> str:
    """A function's arguments after the state, optional ones in brackets."""
    parameters = list(inspect.signature(function).parameters.items())[1:]
    return " ".join(
        name if p.default is inspect.Parameter.empty else f"[{name}={p.default!r}]"
        for name, p in parameters
    )


def _strings(node):
    """Every string in a nested structure of dicts and lists, keys included."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _strings(value)
