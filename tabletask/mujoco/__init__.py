"""The MuJoCo backend: building and sampling a start for a task."""

import numpy as np

from ..language.errors import SamplingError
from ..language.evaluation import evaluate
from ..language.grammar import names
from ..language.schema import Task
from ..robots.base import Robot
from .model import mjcf
from .place import place_all
from .runtime import Runtime

__all__ = ["CAPABILITIES", "PENETRATION", "STARTS", "Runtime", "build"]

CAPABILITIES = ()  # optional observation capabilities this backend provides (none)
STARTS = 12  # placements tried for one concrete task before giving up
PENETRATION = 0.01  # largest allowed overlap between geoms after settling, in metres


def build(task: Task, rng: np.random.Generator, robot=None) -> Runtime:
    """Compile a task into a Runtime and sample a valid, settled start.

    Each draw places the objects, lets them settle, and is rejected if anything
    overlaps, is still moving, left its region, is pinched, or violates the start
    or always conditions.

    Args:
        task: The task to compile.
        rng: Random generator for appearance and placement.
        robot: Optional `Robot` to attach to the scene.

    Returns:
        A Runtime whose current state is the episode start.

    Raises:
        TypeError: If `robot` is not a `Robot`.
        ValueError: If the robot's name clashes with a task object.
        KeyError: If the task names a body, region or joint the scene lacks.
        SamplingError: If no draw out of `STARTS` is valid. The last runtime is
            attached as `error.runtime`.
    """
    runtime = _compile(task, rng, robot)
    rejected = []  # reason each draw was rejected, reported to the task author
    for _ in range(STARTS):
        runtime.reset()
        try:
            place_all(runtime, rng)
        except RuntimeError as error:
            rejected.append(f"placing: {error}")
            continue
        _settle(runtime)
        reason = _rejection(task, runtime)
        if reason is None:
            return runtime
        rejected.append(reason)
    error = _sampling_error(rejected)
    error.runtime = runtime  # the last draw, for inspecting the failure
    raise error


def _compile(task: Task, rng: np.random.Generator, robot) -> Runtime:
    """Check the robot, compile the scene and check every name the task uses."""
    if robot is not None:
        if not isinstance(robot, Robot):
            raise TypeError("robot must be a Robot instance")
        if robot.name in task.scene.objects:
            raise ValueError(f"robot name {robot.name!r} conflicts with a task object")
    runtime = Runtime(task, mjcf(task, rng, robot=robot))
    runtime.robot = robot.bind(runtime) if robot is not None else None
    for name, where in names(task):
        if not runtime.resolves(name):
            raise KeyError(f"{where}: {name!r} is not a body, region or joint in this scene")
    return runtime


def _settle(runtime: Runtime) -> None:
    """Let a placed draw settle and record the settled state as the episode start."""
    runtime.rested = runtime.settle()
    # The settled state, not the placement transient, defines the episode start.
    runtime.spawn_penetration = runtime.penetration()
    runtime.mark_start()


def _rejection(task: Task, runtime: Runtime) -> str | None:
    """The reason a settled draw is rejected, or None if it is a valid start."""
    if runtime.spawn_penetration > PENETRATION:
        return f"objects overlap by {runtime.spawn_penetration * 1000:.0f} mm after settling"
    if not runtime.rested:
        return "still moving after settling"
    moved = [
        name
        for name, obj in task.scene.objects.items()
        if "/" in obj.sample.rel and not runtime.stayed(name, obj.sample.rel)
    ]
    if moved:
        return f"{', '.join(moved)} left the region it was put in"
    pinched = [name for name in task.scene.objects if runtime.pinched(name)]
    if pinched:
        return f"{', '.join(pinched)} pinched between objects"
    if task.scene.start is not None and not evaluate(task.scene.start, runtime):
        parts = (
            task.scene.start.get("and", [task.scene.start])
            if isinstance(task.scene.start, dict)
            else [task.scene.start]
        )
        failed = [str(i + 1) for i, part in enumerate(parts) if not evaluate(part, runtime)]
        return f"start condition {', '.join(failed)} of {len(parts)} is false"
    if not all(evaluate(condition, runtime) for condition in task.goal.always.values()):
        return "an always condition is false"
    return None


def _sampling_error(rejected: list[str]) -> SamplingError:
    """The error raised when no draw is valid, counting each rejection reason."""
    counts = {reason: rejected.count(reason) for reason in dict.fromkeys(rejected)}
    return SamplingError(
        f"no settled start satisfied the start conditions in {STARTS} placements: "
        + "; ".join(f"{count} x {reason}" for reason, count in counts.items())
    )
