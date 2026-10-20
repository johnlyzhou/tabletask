"""Steps that turn a task with variables, counts and defined words into a concrete one."""

from __future__ import annotations

import copy

import msgspec

from .evaluation import unpack_expression
from .kernel import ARITHMETIC
from .schema import Task
from .vocabulary import (
    BOUND,
    OTHER,
    PLACEHOLDER,
    QUANTIFIERS,
    VOCABULARY,
    container_region,
)


def draw(task: Task, rng) -> dict[str, str]:
    """Draw a value for each variable; variables sharing `apart` get distinct values."""
    values: dict[str, str] = {}
    taken: dict[str, set[str]] = {}
    for name, variable in task.variables.items():
        choices = [str(v) for v in variable.one_of]
        if variable.apart is not None:
            used = taken.setdefault(variable.apart, set())
            choices = [c for c in choices if c not in used]
            if not choices:
                raise ValueError(
                    f"variable {name!r} has nothing left to draw: the variables held apart as "
                    f"{variable.apart!r} want more values than {len(variable.one_of)} offers"
                )
        chosen = choices[int(rng.integers(len(choices)))]
        if variable.apart is not None:
            taken[variable.apart].add(chosen)
        values[name] = chosen
    return values


def multiply(task: Task, rng=None) -> Task:
    """Draw each object's count and make numbered copies tagged with its name.

    An object with a count other than 1 always becomes `name1`, `name2`, ..., even when
    a range draws a single copy, so conditions can name the copies reliably.
    """
    if all(obj.count == 1 for obj in task.scene.objects.values()):
        return task

    def copies(record):
        objects = {}
        for name, obj in task.scene.objects.items():
            count = _sample_count(obj.count, rng)
            written = record["scene"]["objects"][name]
            written.pop("count", None)
            if obj.count == 1:
                objects[name] = written
                continue
            for i in range(1, count + 1):
                duplicate = copy.deepcopy(written)
                duplicate["tags"] = sorted({*duplicate.get("tags", []), name})
                objects[f"{name}{i}"] = duplicate
        record["scene"]["objects"] = objects
        return record

    return transform_task(task, copies)


def _sample_count(count, rng) -> int:
    if isinstance(count, int):
        return max(1, count)
    low, high = int(count[0]), int(count[1])
    if rng is None or high <= low:
        return max(1, low)
    return int(rng.integers(low, high + 1))


def bind(task: Task, values: dict[str, str], labels: dict[str, str] | None = None) -> Task:
    """Replace every `{name}` with its drawn value.

    `labels` maps a value to the words the instruction should use for it, so an
    instruction reads "the tin of tuna" where the goal says `tuna_can`.
    """
    if not values:
        return task

    def fill(node, replacements=values):
        if isinstance(node, str):
            return PLACEHOLDER.sub(lambda m: replacements.get(m.group(1), m.group(0)), node)
        if isinstance(node, list):
            return [fill(v, replacements) for v in node]
        if isinstance(node, dict):
            return {fill(k, replacements): fill(v, replacements) for k, v in node.items()}
        return node

    def filled(record):
        instruction = record["instruction"]
        record = fill(record)
        if labels:
            spoken = {name: labels.get(value, value) for name, value in values.items()}
            record["instruction"] = fill(instruction, spoken)
        record["variables"] = {}
        return record

    return transform_task(task, filled)


def group(task: Task, reference: str) -> list[str]:
    """The objects a `#tag` refers to, in scene order."""
    tag = reference[1:]
    members = [name for name, obj in task.scene.objects.items() if tag in obj.tags]
    if not members:
        raise ValueError(f"no object is tagged {tag!r}; a goal about #{tag} would be about nothing")
    return members


def transform_task(task: Task, change) -> Task:
    """A new task from applying `change` to the task's plain-data record."""
    record = msgspec.to_builtins(task, enc_hook=str)
    record.pop("__source__", None)
    result = msgspec.convert(change(record), type=Task)
    result.authored = task.authored
    result.source = task.source
    return result


def _transform_conditions(task: Task, change) -> Task:
    """A new task with `change` applied to every stage, always condition and the start."""

    def every_condition(record):
        for stage in record["goal"]["stages"].values():
            stage["condition"] = change(stage["condition"])
        record["goal"]["always"] = {
            k: change(v) for k, v in record["goal"].get("always", {}).items()
        }
        if record["scene"].get("start") is not None:
            record["scene"]["start"] = change(record["scene"]["start"])
        return record

    return transform_task(task, every_condition)


def inline_states(task: Task) -> Task:
    """Expand the words the task defines for itself (`states`), leaving the vocabulary's."""
    return _expand_definitions(task, task.states) if task.states else task


def inline(task: Task) -> Task:
    """Expand the task's own words, then the standard vocabulary, into kernel words."""
    return _expand_definitions(inline_states(task), VOCABULARY)


def _expand_definitions(task: Task, words: dict) -> Task:
    return _transform_conditions(task, _definitions(words))


def _definitions(words: dict):
    """A function that expands the given words in a condition."""

    def fill(node, bindings):
        if isinstance(node, str):
            if node in bindings:
                return bindings[node]
            owner, slash, region = node.partition("/")
            if slash and isinstance(bindings.get(owner), str):
                return bindings[owner] + "/" + region
            return node
        if isinstance(node, list):
            return [fill(v, bindings) for v in node]
        if not isinstance(node, dict):
            return node
        name = next(iter(node))
        if name in ARITHMETIC:
            left, right = (fill(v, bindings) for v in node[name])
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return ARITHMETIC[name](left, right)
            return {name: [left, right]}
        return {name: fill(node[name], bindings)}

    def apply(word, args):
        definition = words[word]
        bindings = {}
        for i, parameter in enumerate(definition["of"]):
            name, default = (parameter, None) if isinstance(parameter, str) else parameter
            if i < len(args):
                bindings[name] = args[i]
            elif default is not None:
                # a default may name an earlier parameter, as `b` defaults to `a`
                bindings[name] = (
                    bindings.get(default, default) if isinstance(default, str) else default
                )
            else:
                raise ValueError(f"{word!r} needs {name!r}")
        return walk(fill(definition["is"], bindings))

    def walk(node):
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        name, args = unpack_expression(node)
        if name in words:
            return apply(name, [walk(a) for a in container_region(name, list(args))])
        if name in QUANTIFIERS:
            return {name: [args[0], walk(args[1]), *args[2:]]}
        return {name: [walk(a) for a in args]}

    return walk


def expand(task: Task) -> Task:
    """Expand `#tag` groups and quantifiers into conditions on named objects."""
    return _transform_conditions(task, _quantifiers(task))


def _quantifiers(task: Task):
    """A function that expands a task's `#tag` groups and quantifiers in a condition."""

    def rename(node, names: dict):
        if isinstance(node, str):
            owner, slash, region = node.partition("/")
            return names[owner] + slash + region if owner in names else node
        if isinstance(node, list):
            return [rename(v, names) for v in node]
        if isinstance(node, dict):
            return {k: rename(v, names) for k, v in node.items()}
        return node

    def members(reference):
        """A `#tag` names a group; a list names its members directly."""
        return group(task, reference) if str(reference).startswith("#") else list(reference)

    def walk(node):
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        name, args = next(iter(node.items()))
        if name in QUANTIFIERS and isinstance(args, list) and 2 <= len(args) <= 4:
            over, joined = members(args[0]), QUANTIFIERS[name]
            if joined == "pairs":
                first = args[2] if len(args) > 2 else BOUND
                second = args[3] if len(args) > 3 else OTHER
                bodies = [
                    walk(rename(args[1], {first: a, second: b}))
                    for i, a in enumerate(over)
                    for b in over[i + 1 :]
                ]
                return {"and": bodies}
            binder = args[2] if len(args) > 2 else BOUND
            bodies = [walk(rename(args[1], {binder: member})) for member in over]
            if joined == "none":
                return {"not": [{"or": bodies}]} if bodies else {"and": []}
            if joined == "list":
                return bodies
            return {joined: bodies}
        if not isinstance(args, list):
            return {name: walk(args)}
        return {name: [members(a) if str(a).startswith("#") else walk(a) for a in args]}

    return walk


def concrete(task: Task) -> Task:
    """Groups, quantifiers and defined words all expanded, ready to evaluate.

    The stage conditions as they were before their words were expanded are kept in
    `milestones`, for counting progress.
    """
    grouped = expand(task)
    # expand again after inlining: a defined word's body can hold quantifiers too
    result = expand(inline(grouped))
    if not result.milestones:
        result.milestones = {name: s.condition for name, s in grouped.goal.stages.items()}
    return result


def concrete_condition(task: Task, condition):
    """One of a task's conditions expanded as `concrete` expands the task's own."""
    grow = _quantifiers(task)
    condition = grow(condition)
    if task.states:
        condition = _definitions(task.states)(condition)
    return grow(_definitions(VOCABULARY)(condition))
