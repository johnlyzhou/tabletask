"""Static and sampled checks for task documents and asset files."""

from __future__ import annotations

import json
from pathlib import Path

from . import sampling
from .language.evaluation import evaluate
from .language.format import load
from .language.modules import requirements


def check(
    path: Path, build: bool = False, samples: int = 6, seed: int = 0, lint_goal: bool = False
) -> dict:
    """Problems found in a task json or an asset xml.

    Args:
        path: A task document (`.json`) or an asset file (`.xml`).
        build: Also sample task starts in the simulator.
        samples: Number of starts to sample when `build` is set.
        seed: First seed to sample; seeds `seed .. seed + samples - 1` are used.
        lint_goal: Also run `lint` on the task's goal.

    Returns:
        A dict with `path`, `kind` (`"task"` or `"asset"`), `problems` and `notes`,
        plus kind-specific facts. Exceptions raised while checking are recorded as
        problems rather than propagated.
    """
    path = Path(path)
    result = {
        "path": str(path),
        "kind": "asset" if path.suffix == ".xml" else "task",
        "problems": [],
        "notes": [],
    }
    try:
        if result["kind"] == "asset":
            _check_asset(path, result)
        else:
            _check_task(path, result, build, samples, seed, lint_goal)
    except Exception as error:  # report a broken file as a problem instead of crashing
        result["problems"].append(f"{type(error).__name__}: {error}")
    return result


def _check_asset(path: Path, result: dict) -> None:
    from .mujoco.checks import study  # asset files are MuJoCo documents

    facts, problems, _ = study(path)
    result.update(facts)
    result["problems"].extend(problems)


def _check_task(
    path: Path, result: dict, build: bool, samples: int, seed: int, lint_goal: bool = False
) -> None:
    task = load(path)
    result["id"] = task.id
    result["observations"] = sorted(requirements(task))
    if not build:
        return
    result["sampled"] = []
    first = None  # the first start that sampled, for lint
    for episode in range(seed, seed + samples):
        try:
            runtime = sampling.build(task, episode)
        except Exception as error:
            result["problems"].append(f"seed {episode}: {type(error).__name__}: {error}")
            continue
        result["sampled"].append(episode)
        first = first or runtime
        for name, stage in runtime.task.goal.stages.items():
            if evaluate(stage.condition, runtime, {}) and not stage.after:
                result["problems"].append(
                    f"seed {episode}: goal {name!r} is already true at the start"
                )
    if len(result["sampled"]) == samples:
        result["notes"].append(f"{samples} starts sampled from seed {seed}")
    if lint_goal:
        findings = lint(task, first)
        result["problems"].extend(findings)
        if not findings:
            result["notes"].append("lint found nothing")


def _first_stage_conditions(task):
    """The condition of every goal stage that has no `after` dependency."""
    return [stage.condition for stage in task.goal.stages.values() if not stage.after]


def _walk(node):
    """Yield a node and every dict and list nested inside it, depth first."""
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _either_way_up(node) -> str | None:
    """An object an `or` accepts both `upright` and `upside_down`, or None."""
    if not isinstance(node, dict) or "or" not in node or not isinstance(node["or"], list):
        return None
    upright, upside_down = set(), set()
    for branch in node["or"]:
        if not isinstance(branch, dict):
            continue
        for word, args in branch.items():
            if not (isinstance(args, list) and args and isinstance(args[0], str)):
                continue
            if word == "upright":
                upright.add(args[0])
            elif word == "upside_down":
                upside_down.add(args[0])
    both = upright & upside_down
    return sorted(both)[0] if both else None


def _objects_in_region(task, region: str) -> set:
    """The objects the goal places in `region`."""
    found = set()
    for tree in _first_stage_conditions(task):
        for node in _walk(tree):
            if not isinstance(node, dict):
                continue
            for args in node.values():
                if (
                    isinstance(args, list)
                    and len(args) >= 2
                    and args[1] == region
                    and isinstance(args[0], str)
                ):
                    found.add(args[0].split("/")[0])
    return found


def _kept_apart(task, objects: set) -> bool:
    """Whether the goal constrains the relative pose of at least two of `objects`."""
    if len(objects) < 2:
        return False
    for tree in _first_stage_conditions(task):
        for node in _walk(tree):
            if not isinstance(node, dict):
                continue
            for word, args in node.items():
                if word not in ("gap", "spread", "step", "twist"):
                    continue
                named = {name.split("/")[0] for name in _string_args(args)}
                if len(named & objects) >= 2:
                    return True
    return False


def _string_args(args) -> list:
    """The string arguments of an expression, including those one list deep."""
    flat = []
    for arg in args if isinstance(args, list) else []:
        if isinstance(arg, str):
            flat.append(arg)
        elif isinstance(arg, list):
            flat.extend(name for name in arg if isinstance(name, str))
    return flat


def lint(task, runtime=None) -> list[str]:
    """Goal conditions that accept more than the instruction describes.

    Two patterns are flagged: an object accepted either way up, when the instruction
    may depend on which face is up; and a coarse region named where the object offers
    two or more regions under half its volume. The second check needs a runtime and
    is skipped when the goal also names a finer region or keeps the objects apart.

    Args:
        task: The compiled task.
        runtime: A sampled start used to measure regions; if None, only the first
            check runs.

    Returns:
        One message per finding; sorted and de-duplicated when `runtime` is given.
    """
    found = _either_way_up_findings(task)
    if runtime is None:
        return found
    found += _coarse_region_findings(task, runtime)
    return sorted(set(found))


def _either_way_up_findings(task) -> list[str]:
    """One message per goal `or` that accepts an object either way up."""
    found = []
    for tree in _first_stage_conditions(task):
        for node in _walk(tree):
            name = _either_way_up(node)
            if name:
                found.append(f"{name!r} counts either way up; one face may be all that matters")
    return found


def _coarse_region_findings(task, runtime) -> list[str]:
    """One message per object region the goal names where finer regions are offered."""
    found = []
    written = " ".join(json.dumps(tree) for tree in _first_stage_conditions(task))
    for tree in _first_stage_conditions(task):
        for node in _walk(tree):
            if not isinstance(node, dict):
                continue
            for args in node.values():
                if not (isinstance(args, list) and len(args) >= 2):
                    continue
                named = args[1]
                if not isinstance(named, str) or runtime.region(named) is None:
                    continue
                owner = named.split("/")[0]
                if owner not in runtime.task.scene.objects:
                    continue  # table regions are coarse by design
                finer = _finer_regions(runtime, owner, named)
                if finer is None:
                    continue
                # If the goal already names a finer region, the coarse one is deliberate.
                if any(f'"{owner}/{name}"' in written for name in finer):
                    continue
                if _kept_apart(task, _objects_in_region(task, named)):
                    continue
                if len(finer) >= 2:
                    found.append(
                        f"{named} is the coarse choice; {owner} also offers "
                        + ", ".join(sorted(finer)[:6])
                    )
    return found


def _finer_regions(runtime, owner: str, named: str) -> list[str] | None:
    """Short names of `owner`'s regions under half the volume of `named`, or None."""
    try:
        lo, hi = runtime.aabb(named)
    except Exception:
        return None
    volume = float((hi - lo).prod())
    finer = []
    for other in runtime.places():
        if not other.startswith(owner + "/") or other == named:
            continue
        try:
            other_lo, other_hi = runtime.aabb(other)
        except Exception:
            continue
        if float((other_hi - other_lo).prod()) < volume * 0.5:
            finer.append(other.split("/")[-1])
    return finer


def summary(result: dict) -> str:
    """A one-line summary of a `check` result, plus one line per problem."""
    name = Path(result["path"]).stem
    if result["problems"]:
        lines = [f"{name}: {len(result['problems'])} problem(s)"]
        lines.extend(f"  {problem}" for problem in result["problems"])
        return "\n".join(lines)

    details = list(result["notes"])
    if result["kind"] == "task":
        observations = ", ".join(result["observations"]) or "core geometry"
        details.append(f"observations: {observations}")
    if details:
        return f"{name}: ok ({', '.join(details)})"
    return f"{name}: ok"
